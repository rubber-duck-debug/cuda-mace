#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void wmma_dense(torch::Tensor U, torch::Tensor W, torch::Tensor C);
void wmma_dense_multiwarp(torch::Tensor U, torch::Tensor W, torch::Tensor C);
void UwN_contraction(torch::Tensor Uw, torch::Tensor N, torch::Tensor C);
void UwN_contraction_sparse(torch::Tensor Uw, torch::Tensor Uw_indexes, torch::Tensor features, torch::Tensor C);

void UwN_contraction_sparse_new(torch::Tensor Uw_dense, torch::Tensor Uw_indexes, torch::Tensor Uw_nvals, torch::Tensor features, torch::Tensor C,
		const int nthreadsx, const int nthreadsy, const int nblocksx, const int nblocksy);

torch::Tensor get_wmma_dense(torch::Tensor U, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	wmma_dense(U, W, output);

	return output;
}

torch::Tensor get_wmma_dense_multiwarp(torch::Tensor U, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	wmma_dense_multiwarp(U, W, output);

	return output;
}

torch::Tensor get_wmma_UwN_dense(torch::Tensor Uw, torch::Tensor N) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { Uw.size(0), Uw.size(1), N.size(0), N.size(1) }, options);

	UwN_contraction(Uw, N, output);

	return output;
}

torch::Tensor get_UwN_sparse(torch::Tensor Uw, torch::Tensor U_indexes, torch::Tensor features) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { Uw.size(0), Uw.size(1), features.size(1), features.size(0) }, options);

	UwN_contraction_sparse(Uw, U_indexes, features, output);

	return output;
}

torch::Tensor get_UwN_sparse_new(torch::Tensor Uw, torch::Tensor U_indexes, torch::Tensor nvals, torch::Tensor features, const int nthreadsx,
		const int nthreadsy, const int nblocksx, const int nblocksy) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { features.size(0), features.size(2), Uw.size(0), Uw.size(1) }, options);

	UwN_contraction_sparse_new(Uw, U_indexes, nvals, features, output, nthreadsx, nthreadsy, nblocksx, nblocksy);

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_wmma_dense", &get_wmma_dense, "");
	m.def("get_wmma_dense_multiwarp", &get_wmma_dense_multiwarp, "");
	m.def("get_wmma_UwN_dense", &get_wmma_UwN_dense, "");
	m.def("get_UwN_sparse", &get_UwN_sparse, "");
	m.def("get_UwN_sparse_new", &get_UwN_sparse_new, "");

}
