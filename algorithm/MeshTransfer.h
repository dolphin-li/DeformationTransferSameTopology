#pragma once
#include <vector>
#include <eigen\Dense>
#include <eigen\Sparse>

// Deformation Transfer
// Input:
//	A0, A1, ..., An
//  B0
// Output:
//	  , B1, ..., Bn
// Where Ai, Bi are triangle meshes with the same topology
//	i.e., same faces and the same number of vertices.

/// TODO: we should manually specify anchor points, instead of just fixing the 0th vertices.

class MeshTransfer
{
public:
	typedef double real;
	typedef Eigen::Matrix<real, -1, 1> Vec;
	typedef Eigen::Matrix<real, -1, -1> Mat;
	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> SpMat;
	typedef Eigen::Vector3f Float3;
	typedef Eigen::Matrix3f Mat3f;
	typedef Eigen::Vector3i Int3;
	typedef Eigen::Vector4i Int4;
public:
	MeshTransfer() {}
	~MeshTransfer() {}

	// Initialize the topology and 0th reference mesh A0
	bool init(int nTriangles, const Int3* pTriangles, int nVertices, const Float3* pSrcVertices0);

	// Given B0, Ai, output Bi
	bool transfer(const std::vector<Float3>& tarVerts0, const std::vector<Float3>& srcVerts1, std::vector<Float3>& tarVerts1);

	const char* getErrString()const;
protected:
	void clear();
	void findAnchorPoints();

	void setup_E1MatAndRhs(const std::vector<Float3>& srcVertsDeformed, const std::vector<Float3>& tarVerts0);
	void setup_ancorMat();
	void setup_ancorRhs(const std::vector<Float3>& tarVerts0);
	void setup_RegularizationMat();
	void setup_RegularizationRhs(const std::vector<Float3>& tarVerts0);

	void vertex_vec_to_point(const Vec& x, std::vector<Float3>& verts)const;
	void vertex_point_to_vec(Vec& x, const std::vector<Float3>& verts, const std::vector<Int3>& faces)const;
private:
	bool m_bInit = false;
	std::vector<Int3> m_facesTri;		// triangles converted from src mesh
	std::vector<int> m_anchors;			// index of all anchor points
	std::vector<Float3> m_srcVerts0;
	std::string m_errStr;

	// energy related
	SpMat m_E1Mat, m_E1MatT, m_E1TE1;	// the energy for src-tar triangle correspondences
	Vec m_E1Rhs;						// the energy for src-tar triangle correspondences

	SpMat m_ancorMat;					// for anchor points
	Vec m_ancorRhs;						// for anchor points
	SpMat m_regAtA;						// for isolated-point regularization
	Vec m_regAtb;						// for isolated-point regularization
	SpMat m_ancorRegSumAtA;
	Vec m_ancorRegSumAtb;

	SpMat m_AtA;						// the total energy matrix
	Vec m_Atb, m_x;						// the total right-hand-side value and the solved result
	Eigen::SimplicialCholesky<SpMat> m_solver;
	bool m_shouldAnalysisTopology = false;
};
