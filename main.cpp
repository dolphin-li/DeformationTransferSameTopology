// DeformationTransferSameTopology.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include "MeshTransfer.h"
#include "Renderable\ObjMesh.h"

static void objMeshGetFace(const ObjMesh& mesh, std::vector<MeshTransfer::Int3>& triangles)
{
	triangles.clear();
	for (size_t iFace = 0; iFace < mesh.face_list.size(); iFace++)
	{
		const ObjMesh::obj_face& f = mesh.face_list[iFace];
		for (int k = 0; k < f.vertex_count - 2; k++)
			triangles.push_back(MeshTransfer::Int3(f.vertex_index[0], f.vertex_index[k+1], f.vertex_index[k+2]));
	}
}

static void objMeshGetVerts(const ObjMesh& mesh, std::vector<MeshTransfer::Float3>& verts)
{
	verts.resize(mesh.vertex_list.size());
	for (size_t iVert = 0; iVert < mesh.vertex_list.size(); iVert++)
		verts[iVert] = MeshTransfer::Float3(mesh.vertex_list[iVert][0], mesh.vertex_list[iVert][1], mesh.vertex_list[iVert][2]);
}

static void objMeshSetVerts(ObjMesh& mesh, const std::vector<MeshTransfer::Float3>& verts)
{
	mesh.vertex_list.resize(verts.size());
	for (size_t iVert = 0; iVert < mesh.vertex_list.size(); iVert++)
		mesh.vertex_list[iVert] = ldp::Float3(verts[iVert][0], verts[iVert][1], verts[iVert][2]);
}

int main(int argc, const char* argv[])
{
	if (argc != 3 && argc != 4)
	{
		printf("Usage: dtransfer.exe [src_folder] [target0.obj] [result_folder]");
		return -1;
	}
	ObjMesh srcMesh0, tarMesh0;
	std::vector<MeshTransfer::Float3> srcVerts0, srcVerts1, tarVerts0;
	std::vector<MeshTransfer::Int3> triangles;

	std::string src_folder("mean/");
	std::string result_folder("");
	if (src_folder.back() != '/' && src_folder.back() != '\\')
		src_folder.append("/");
	if (argc == 4)
		result_folder = argv[3];
	if (result_folder != "")
	{
		if (result_folder.back() != '/' && result_folder.back() != '\\')
			result_folder.append("/");
	}

#ifdef _WIN32
	std::string win_result_folder = result_folder;
	for (auto& c : win_result_folder)
	{
		if (c == '/')
			c = '\\';
	}
	DWORD dwAttrib = GetFileAttributesA(win_result_folder.c_str());

	if (!(dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY)))
	{
		char cmd[1024];
		sprintf_s(cmd, "mkdir %s", win_result_folder.c_str());
		printf("%s\n", cmd);
		system(cmd);
	}
#endif

	if (!srcMesh0.loadObj((src_folder + "0.obj").c_str(), false, false))
	{
		printf("Error, source mesh not found: %s\n", (src_folder + "0.obj").c_str());
		return -1;
	}
	if (!tarMesh0.loadObj(argv[2], false, false))
	{
		printf("Error, target mesh not found: %s\n", argv[2]);
		return -1;
	}

	objMeshGetFace(srcMesh0, triangles);
	objMeshGetVerts(srcMesh0, srcVerts0);
	objMeshGetVerts(tarMesh0, tarVerts0);

	enum {NUM_THREADS = 4};
	MeshTransfer transfer[NUM_THREADS];
		
	for (int i = 0; i < NUM_THREADS; i++)
	{
		if (!transfer[i].init((int)triangles.size(), triangles.data(), (int)srcVerts0.size(), srcVerts0.data()))
		{
			printf("[thread=%d]: %s\n", i, transfer[i].getErrString());
			return -1;
		}
	}

#pragma omp parallel for num_threads(NUM_THREADS)
	for (int iMesh = 0; iMesh < 47; iMesh++)
	{
		const int tid = omp_get_thread_num();

		ObjMesh tarMesh1, srcMesh1;
		std::vector<MeshTransfer::Float3> tarVerts1;
		tarMesh1.cloneFrom(&tarMesh0);

		std::string sourceMeshName(src_folder + std::to_string(iMesh) + ".obj");
		if (!srcMesh1.loadObj(sourceMeshName.c_str(), false, false))
		{
			printf("warning, source mesh not found: %s\n", sourceMeshName.c_str());
			continue;
		}

		gtime_t tbegin = ldp::gtime_now();

		objMeshGetVerts(srcMesh1, srcVerts1);
		if (!transfer[tid].transfer(tarVerts0, srcVerts1, tarVerts1))
		{
			printf("%s\n", transfer[tid].getErrString());
			continue;
		}
		objMeshSetVerts(tarMesh1, tarVerts1);

		gtime_t tend = ldp::gtime_now();
		printf("transfer time[%d]: %f sec\n", iMesh, ldp::gtime_seconds(tbegin, tend));

		tarMesh1.saveObj((result_folder + std::to_string(iMesh) + ".obj").c_str());
	}

    return 0;
}

