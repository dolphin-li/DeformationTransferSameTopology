#pragma once


/**
* Parameters for Mesh Based Deformation Transfer
*/
namespace MeshTransferParameter
{
	// anchor points are automatically decided by the boundary points of markered components.
#define USE_BOUNDARY_AS_ANCHOR

	/**
	* Parameters for Deformation Transfer
	* Transfer Energy Function:
	*	min || Ws * Es + Wi * Ei + Wc * Ec + Wa * Ea + W1 * E1 ||^2
	*		Es: the smoothness term for the target mesh, 
	*				ONLY for target trangle that cannot find corresponding src traingle.
	*		Ei: the identity term for the target mesh
	*				ONLY for target trangle that cannot find corresponding src traingle.
	*		Ea: the anchor points energy
	*				Now the anchor points are automatically selected via the boundaries of the markered components
	*		E1: the correspond energy, deformation gradient related
	* Note:
	*	Each of the weight will be divided by the number of constraints and then applied.
	*/
	const static double Transfer_Weight_Smoothness = 1e-3;
	const static double Transfer_Weight_Identity = 1e-6;
	const static double Transfer_Weight_Correspond = 1.0;
	const static double Transfer_Weight_Anchor = 1e8;
	const static double Transfer_Weight_Regularization = 1e-8;

	// from Li Hao's "Example-Based Facial Rigging"
	// In transfering, the weight of each triangle is not the same:
	// if the gradient of a triangle moves little in the src pair, 
	// it should be emphasised to move a little in the target pair
	// The weight is calculated via:
	//	(1 + ||M||_F)^theta / (kappa + ||M||_F)^theta
	const static double Transfer_Graident_Emhasis_kappa = 0.1;
	const static double Transfer_Graident_Emhasis_theta = 2.5;
};