#include "MeshTransfer.h"
#include <fstream>
#include "ParameterConfig.h"

typedef MeshTransfer::real real;
typedef MeshTransfer::Float3 Float3;
typedef MeshTransfer::Int3 Int3;
typedef MeshTransfer::Int4 Int4;
typedef MeshTransfer::Mat3f Mat3f;

template <class T>
static bool hasIllegalData(const T* data, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (std::isinf(data[i]) || std::isnan(data[i]))
			return true;
	}
	return false;
}

bool hasIllegalTriangle(const Int3* pTris, int n)
{
	for (int i = 0; i < n; i++)
	{
		const Int3& t = pTris[i];
		if (t[0] < 0 || t[1] < 0 || t[2] < 0)
			return true;
		if (t[0] == t[1] || t[0] == t[2] || t[1] == t[2])
			return true;
	}
	return false;
}

const char* MeshTransfer::getErrString()const
{
	return m_errStr.c_str();
}

bool MeshTransfer::init(int nTriangles, const Int3* pTriangles, int nVertices, const Float3* pSrcVertices0)
{
	clear();
	if (hasIllegalData((const float*)pSrcVertices0, nVertices * 3))
	{
		m_errStr = "nan or inf in input pSrcVertices0";
		return false;
	}
	if (hasIllegalTriangle(pTriangles, nTriangles))
	{
		m_errStr = "illegal or trivial triangles in pTriangles!";
		return false;
	}

	m_srcVerts0.resize(nVertices);
	for (int i = 0; i < nVertices; i++)
		m_srcVerts0[i] = pSrcVertices0[i];

	m_facesTri.resize(nTriangles);
	for (int i = 0; i < nTriangles; i++)
		m_facesTri[i] = pTriangles[i];

	// precomputation
	findAnchorPoints();
	setup_ancorMat();
	setup_RegularizationMat();
	const real w_anchor = real(MeshTransferParameter::Transfer_Weight_Anchor / (1e-3f + m_ancorMat.rows()));
	const real w_reg = real(MeshTransferParameter::Transfer_Weight_Regularization / (1e-3f + m_regAtA.rows()));
	m_ancorRegSumAtA = m_ancorMat.transpose() * m_ancorMat * w_anchor + m_regAtA * w_reg;

	m_bInit = true;
	m_shouldAnalysisTopology = true;
	return true;
}

template<class T>
static void FastTransGivenStructure(const Eigen::SparseMatrix<T>& A, Eigen::SparseMatrix<T>& At)
{
	Eigen::VectorXi positions(At.outerSize());
	for (int i = 0; i<At.outerSize(); i++)
		positions[i] = At.outerIndexPtr()[i];
	for (int j = 0; j<A.outerSize(); ++j)
	{
		for (Eigen::SparseMatrix<T>::InnerIterator it(A, j); it; ++it)
		{
			int i = it.index();
			int pos = positions[i]++;
			At.valuePtr()[pos] = it.value();
		}
	}
}

template<class T>
void FastAtAGivenStructure(const Eigen::SparseMatrix<T>& A, const Eigen::SparseMatrix<T>& At, Eigen::SparseMatrix<T>& AtA)
{
	MeshTransfer::Vec Tmp;
	Eigen::VectorXi Mark;
	Tmp.resize(AtA.innerSize());
	Mark.resize(AtA.innerSize());
	Mark.setZero();

	for (int j = 0; j<AtA.outerSize(); j++)
	{
		for (Eigen::SparseMatrix<T>::InnerIterator it_A(A, j); it_A; ++it_A)
		{
			int k = it_A.index();
			real v_A = it_A.value();

			for (Eigen::SparseMatrix<T>::InnerIterator it_At(At, k); it_At; ++it_At)
			{
				int i = it_At.index();
				real v_At = it_At.value();
				if (!Mark[i])
				{
					Mark[i] = 1;
					Tmp[i] = v_A*v_At;
				}
				else
					Tmp[i] += v_A*v_At;
			}//end for it_At
		}//end for it_A

		for (Eigen::SparseMatrix<T>::InnerIterator it(AtA, j); it; ++it)
		{
			int i = it.index();
			it.valueRef() = Tmp[i];
			Mark[i] = 0;
		}
	}//end for i
}

bool MeshTransfer::transfer(const std::vector<Float3>& tarVerts0,
	const std::vector<Float3>& srcVertsDeformed, std::vector<Float3>& tarVertsDeformed)
{
	if (!m_bInit)
	{
		m_errStr = "not initialized when calling transfer()";
		return false;
	}
	if (tarVerts0.size() != m_srcVerts0.size() || srcVertsDeformed.size() != m_srcVerts0.size())
	{
		m_errStr = "transfer: vertex size not matched!";
		return false;
	}
	if (hasIllegalData((const float*)tarVerts0.data(), (int)tarVerts0.size() * 3))
	{
		m_errStr = "nan or inf in tarVerts0!";
		return false;
	}
	if (hasIllegalData((const float*)srcVertsDeformed.data(), (int)srcVertsDeformed.size() * 3))
	{
		m_errStr = "nan or inf in srcVertsDeformed!";
		return false;
	}

	// computing all energy matrices
	setup_E1MatAndRhs(srcVertsDeformed, tarVerts0);
	setup_ancorRhs(tarVerts0);
	setup_RegularizationRhs(tarVerts0);

	// all energy weights should be normalized by their number of terms
	const real w1 = real(MeshTransferParameter::Transfer_Weight_Correspond / (1e-3f + m_E1Mat.rows()));
	const real w_anchor = real(MeshTransferParameter::Transfer_Weight_Anchor / (1e-3f + m_ancorMat.rows()));
	const real w_reg = real(MeshTransferParameter::Transfer_Weight_Regularization / (1e-3f + m_regAtA.rows()));

	if (hasIllegalData(m_E1Mat.valuePtr(), (int)m_E1Mat.nonZeros()))
	{
		m_errStr = "nan or inf in E1Mat!";
		return false;
	}

	// sum all the energy terms
	if (m_shouldAnalysisTopology)
	{
		m_E1MatT = m_E1Mat.transpose();
		m_E1TE1 = m_E1MatT * m_E1Mat;
		m_AtA = m_E1TE1 * w1 + m_ancorRegSumAtA;
	}
	else
	{
		FastTransGivenStructure(m_E1Mat, m_E1MatT); 
		FastAtAGivenStructure(m_E1Mat, m_E1MatT, m_E1TE1);
		m_AtA = m_E1TE1 * w1 + m_ancorRegSumAtA;
	}
	m_ancorRegSumAtb = m_ancorMat.transpose() * m_ancorRhs * w_anchor + m_regAtb * w_reg;
	m_Atb = m_E1MatT * m_E1Rhs * w1 + m_ancorRegSumAtb;
	
	// solve
	if (m_shouldAnalysisTopology)
	{
		m_solver.analyzePattern(m_AtA);
		m_shouldAnalysisTopology = false;
	}
	m_solver.factorize(m_AtA);
	m_x = m_solver.solve(m_Atb);

	// return the value
	vertex_vec_to_point(m_x, tarVertsDeformed);

	if (hasIllegalData((const float*)tarVertsDeformed.data(), (int)tarVertsDeformed.size() * 3))
	{
		m_errStr = "finished transfer, but nan or inf in tarVertsDeformed!";
		return false;
	}
	return true;
}

void MeshTransfer::clear()
{
	m_bInit = false;
	m_shouldAnalysisTopology = false;
	m_facesTri.clear();
	m_anchors.clear();
	m_srcVerts0.clear();
}

void MeshTransfer::findAnchorPoints()
{
	// ldp: may be we should manually specify anchor points
	//		to make the results' global position reasonable
	m_anchors.clear();
	m_anchors.push_back(0);
}

void MeshTransfer::vertex_vec_to_point(const Vec& x, std::vector<Float3>& verts)const
{
	verts.resize(m_srcVerts0.size());
	for (int i = 0; i < verts.size(); i++)
		for (int k = 0; k < 3; k++)
			verts[i][k] = (float)x[k*(x.size() / 3) + i];
}

void MeshTransfer::vertex_point_to_vec(Vec& x, const std::vector<Float3>& verts, const std::vector<Int3>& faces)const
{
	int nTotalVerts = (int)verts.size() + (int)faces.size();
	if (x.size() != nTotalVerts)
		x.resize(nTotalVerts * 3);

	x.setZero();
	for (int i = 0; i < verts.size(); i++)
		for (int k = 0; k < 3; k++)
			x[k*nTotalVerts + i] = verts[i][k];

	for (int i = 0; i < faces.size(); i++)
	{
		int row = (int)verts.size() + i;
		const Int3& f = faces[i];
		Float3 v = verts[f[0]] + (verts[f[1]] - verts[f[0]]).cross(verts[f[2]] - verts[f[0]]).normalized();
		for (int k = 0; k < 3; k++)
			x[k*nTotalVerts + row] = v[k];
	}
}

inline void fill4VertsOfFace(int id_f, const std::vector<Int3>& faces, const std::vector<Float3>& verts, int *id_v, Float3* v)
{
	Int3 f = faces[id_f];
	int k = 0;
	for (int k = 0; k < 3; k++)
	{
		id_v[k] = f[k];
		v[k] = verts[id_v[k]];
	}
	v[3] = v[0] + (v[1] - v[0]).cross(v[2] - v[0]).normalized();
	id_v[3] = (int)verts.size() + id_f;
}

inline Mat3f getV(Float3* v)
{
	Mat3f V;
	for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
			V(y, x) = v[x + 1][y] - v[0][y];
	return V;
}

inline Eigen::Matrix<real, 9, 12> getMatrix_namedby_T(Float3* v)
{
	Eigen::Matrix<real, 9, 12> T = Eigen::Matrix<real, 9, 12>::Zero();
	Mat3f V = getV(v).inverse();

	// The matrix T is in block diag style:
	// | A 0 0 |
	// | 0 A 0 |
	// | 0 0 A |
	// where each A is a 3x4 matrix
	for (int y = 0; y < 9; y += 3)
	{
		int x = y / 3 * 4;
		T(y + 0, x + 0) = -V(0, 0) - V(1, 0) - V(2, 0);
		T(y + 0, x + 1) = V(0, 0);
		T(y + 0, x + 2) = V(1, 0);
		T(y + 0, x + 3) = V(2, 0);
		T(y + 1, x + 0) = -V(0, 1) - V(1, 1) - V(2, 1);
		T(y + 1, x + 1) = V(0, 1);
		T(y + 1, x + 2) = V(1, 1);
		T(y + 1, x + 3) = V(2, 1);
		T(y + 2, x + 0) = -V(0, 2) - V(1, 2) - V(2, 2);
		T(y + 2, x + 1) = V(0, 2);
		T(y + 2, x + 2) = V(1, 2);
		T(y + 2, x + 3) = V(2, 2);
	}
	return T;
}

inline void fillCooSys_by_Mat(std::vector<Eigen::Triplet<real>>& cooSys, int row,
	int nTotalVerts, int* id, const Eigen::Matrix<real, 9, 12, 0, 9, 12>& T)
{
	// The matrix T is in block diag style:
	// | A 0 0 |
	// | 0 A 0 |
	// | 0 0 A |
	// where each A is a 3x4 matrix
	const static int nBlocks = 3;
	const static int nPoints = 4;
	const static int nCoords = 3;
	int pos = row * 4;
	for (int iBlock = 0; iBlock < nBlocks; iBlock++)
	{
		const int xb = iBlock * nPoints;
		const int yb = iBlock * nCoords;
		for (int y = 0; y < nCoords; y++)
		{
			for (int x = 0; x < nPoints; x++)
			{
				const int col = nTotalVerts * iBlock + id[x];
				cooSys[pos++] = Eigen::Triplet<real>(row + yb + y, col, T(yb + y, xb + x));
			}
		}
	} // end for iBlock
}

void MeshTransfer::setup_E1MatAndRhs(const std::vector<Float3>& srcVertsDeformed, const std::vector<Float3>& tarVerts0)
{
	const int nMeshVerts = (int)tarVerts0.size();
	const int nTotalVerts = nMeshVerts + (int)m_facesTri.size();
	std::vector<Eigen::Triplet<real>> cooSys;

	m_E1Rhs.resize(m_facesTri.size() * 9);
	cooSys.resize(m_E1Rhs.size() * 4);

	for (int iFace = 0; iFace < (int)m_facesTri.size(); iFace++)
	{
		// face_i_tar
		Int4 id_vi_tar;
		Float3 vi_tar[4];
		fill4VertsOfFace(iFace, m_facesTri, tarVerts0, id_vi_tar.data(), vi_tar);
		Eigen::Matrix<real, 9, 12> Ti = getMatrix_namedby_T(vi_tar);

		// face_i_src
		Int4 id_vi_src0, id_vi_src1;
		Float3 vi_src0[4], vi_src1[4];
		fill4VertsOfFace(iFace, m_facesTri, m_srcVerts0, id_vi_src0.data(), vi_src0);
		fill4VertsOfFace(iFace, m_facesTri, srcVertsDeformed, id_vi_src1.data(), vi_src1);

		// calculate the weightings
		Mat3f V_src_0 = getV(vi_src0);
		Mat3f V_src_1 = getV(vi_src1);
		Mat3f G_src = V_src_1 * V_src_0.inverse();
		float G_src_norm = (G_src - Mat3f::Identity()).norm();
		real weight_tri =
			pow(
				real((1.f + G_src_norm) / (MeshTransferParameter::Transfer_Graident_Emhasis_kappa + G_src_norm)),
				real(MeshTransferParameter::Transfer_Graident_Emhasis_theta)
			);
		weight_tri = sqrt(weight_tri);

		// construct the gradient transfer matrix
		Eigen::Matrix<real, 9, 12> Si_A = getMatrix_namedby_T(vi_src0);
		Eigen::Matrix<real, 12, 1> Si_x;
		bool inValid = std::isnan(weight_tri) || std::isinf(weight_tri)
			|| hasIllegalData(Ti.data(), Ti.size()) || hasIllegalData(Si_A.data(), Si_A.size());
		for (int k = 0; k < 4; k++)
		{
			for (int kk = 0; kk < 3; kk++)
				Si_x[4 * kk + k] = vi_src1[k][kk];
		}
		if (inValid)
		{
			Si_A.setZero();
			Si_x.setZero();
			Ti.setZero();
		}
		Eigen::Matrix<real, 9, 1> Si_b = Si_A * Si_x;

		// push matrix
		const int row = iFace * 9;
		fillCooSys_by_Mat(cooSys, row, nTotalVerts, id_vi_tar.data(), weight_tri * Ti);
		for (int y = 0; y < Ti.rows(); y++)
			m_E1Rhs[row + y] = weight_tri * Si_b[y];
	}


	m_E1Mat.resize((int)m_E1Rhs.size(), nTotalVerts * 3);
	if (cooSys.size() > 0)
		m_E1Mat.setFromTriplets(cooSys.begin(), cooSys.end());
}

void MeshTransfer::setup_ancorMat()
{
	const int nMeshVerts = (int)m_srcVerts0.size();
	const int nTotalVerts = nMeshVerts + (int)m_facesTri.size();
	m_ancorMat.resize((int)m_anchors.size() * 3, nTotalVerts * 3);

	// build matrix
	for (int i = 0; i < m_anchors.size(); i++)
	{
		for (int k = 0; k<3; k++)
			m_ancorMat.insert(i * 3 + k, m_anchors[i] + nTotalVerts*k) = 1;
	}
	m_ancorMat.finalize();
}

void MeshTransfer::setup_ancorRhs(const std::vector<Float3>& tarVerts0)
{
	const int nMeshVerts = (int)tarVerts0.size();
	const int nTotalVerts = nMeshVerts + (int)m_facesTri.size();
	m_ancorRhs.resize((int)m_anchors.size() * 3);
	m_ancorRhs.setZero();

	// build matrix
	for (int i = 0; i < m_anchors.size(); i++)
	{
		for (int k = 0; k<3; k++)
			m_ancorRhs[i * 3 + k] = tarVerts0[m_anchors[i]][k];
	}
}

void MeshTransfer::setup_RegularizationMat()
{
	const int nMeshVerts = (int)m_srcVerts0.size();
	const int nTotalVerts = nMeshVerts + (int)m_facesTri.size();
	m_regAtA.resize(nTotalVerts * 3, nTotalVerts * 3);
	m_regAtA.reserve(nTotalVerts * 3);
	for (int row = 0; row < m_regAtA.rows(); row++)
		m_regAtA.insert(row, row) = 1;
	m_regAtA.finalize();
}

void MeshTransfer::setup_RegularizationRhs(const std::vector<Float3>& tarVerts0)
{
	const int nMeshVerts = (int)m_srcVerts0.size();
	const int nTotalVerts = nMeshVerts + (int)m_facesTri.size();
	m_regAtb.resize(nTotalVerts * 3);
	m_regAtb.setZero();
	for (int iVert = 0; iVert < nMeshVerts; iVert++)
	{
		for (int k = 0; k < 3; k++)
			m_regAtb[iVert + k * nTotalVerts] = tarVerts0[iVert][k];
	}
}