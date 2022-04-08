#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void Uw3_dense_contraction_tensorcore(torch::Tensor U, torch::Tensor W,
		torch::Tensor C);
void Uw3_wmma_sparse_contraction_tensorcore_kernel_16x16_f32(
		torch::Tensor U_values, torch::Tensor U_indices, torch::Tensor U_nvals,
		torch::Tensor Uw_nvals, torch::Tensor W, torch::Tensor C);
void Uw3_sparse_contraction_kernel_16x16_f32(torch::Tensor U_values,
		torch::Tensor U_indices, torch::Tensor U_nvals,
		torch::Tensor Uw_indices, torch::Tensor Uw_nvals, torch::Tensor W,
		torch::Tensor C);

void UwN3_sparse_contraction(torch::Tensor Uw_dense, torch::Tensor Uw_indexes,
		torch::Tensor Uw_nvals, torch::Tensor features, torch::Tensor C,
		const int nblocksx, const int nblocksy, const int nblocksz,
		const int nthreadsx, const int nthreadsy);

void UwN2_dense_contraction(torch::Tensor Uw3_dense, torch::Tensor features,
		torch::Tensor C, const int nblocksx, const int nblocksy,
		const int nthreadsx, const int nthreadsy);

void multiwarp_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C);

torch::Tensor get_multiwarp_matmul(torch::Tensor A, torch::Tensor B) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { A.size(0), B.size(1) }, options);

	multiwarp_matmul(A, B, output);

	return output;
}

torch::Tensor get_Uw3_dense_tensorcore(torch::Tensor U, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros(
			{ U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	Uw3_dense_contraction_tensorcore(U, W, output);

	return output;
}

torch::Tensor get_Uw3_sparse_contraction_tc(torch::Tensor U_values,
		torch::Tensor U_indices, torch::Tensor U_nvals, torch::Tensor Uw_nvals,
		torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U_values.size(0), U_values.size(1),
			U_values.size(2), W.size(1) }, options);

	Uw3_wmma_sparse_contraction_tensorcore_kernel_16x16_f32(U_values, U_indices,
			U_nvals, Uw_nvals, W, output);

	return output;
}

torch::Tensor get_Uw3_sparse_contraction(torch::Tensor U_values,
		torch::Tensor U_indices, torch::Tensor U_nvals,
		torch::Tensor Uw_indices, torch::Tensor Uw_nvals, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U_values.size(0), U_values.size(1),
			U_values.size(2), W.size(1) }, options);

	Uw3_sparse_contraction_kernel_16x16_f32(U_values, U_indices, U_nvals,
			Uw_indices, Uw_nvals, W, output);

	return output;
}

torch::Tensor get_UwN3_sparse(torch::Tensor Uw, torch::Tensor U_indexes,
		torch::Tensor nvals, torch::Tensor features, const int nblocksx,
		const int nblocksy, const int nblocksz, const int nthreadsx,
		const int nthreadsy) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	int natoms = features.size(0);
	int nirreps = features.size(1);
	int nfeatures = features.size(2);

	//node features: node_feats = torch.randn(num_atoms , num_irreps, num_features, device='cuda')

	//natoms, nirreps, nirreps, nfeatures

	torch::Tensor output = torch::zeros(
			{ natoms, nirreps, nirreps, nfeatures }, options);

	UwN3_sparse_contraction(Uw, U_indexes, nvals, features, output, nblocksx,
			nblocksy, nblocksz, nthreadsx, nthreadsy);

	return output;
}

torch::Tensor get_UwN2_dense_contraction(torch::Tensor Uw2,
		torch::Tensor features, const int nblocksx, const int nblocksy,
		const int nthreadsx, const int nthreadsy) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	int natoms = features.size(0);
	int nfeatures = features.size(1);
	int nirreps = features.size(2);

	torch::Tensor output = torch::zeros( { natoms, nfeatures, nirreps },
			options);

	UwN2_dense_contraction(Uw2, features, output, nblocksx, nblocksy, nthreadsx,
			nthreadsy);

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_Uw3_dense_tensorcore", &get_Uw3_dense_tensorcore, "");
	m.def("get_Uw3_sparse_contraction_tc", &get_Uw3_sparse_contraction_tc, "");
	m.def("get_Uw3_sparse_contraction", &get_Uw3_sparse_contraction, "");
	m.def("get_UwN3_sparse", &get_UwN3_sparse, "");
	m.def("get_UwN2_dense_contraction", &get_UwN2_dense_contraction, "");
}
