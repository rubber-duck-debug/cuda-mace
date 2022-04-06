#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void DenseTensor3Contraction_fp32(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void DenseTensor3Contraction_fp64(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void SparseTensor3Contraction_fp32(torch::Tensor nnon_zero, torch::Tensor indices, torch::Tensor U, torch::Tensor W, torch::Tensor C);
void SparseTensor3Contraction_fp64(torch::Tensor nnon_zero, torch::Tensor indices, torch::Tensor U, torch::Tensor W, torch::Tensor C);

void WMMATensor3Contraction_fp32(torch::Tensor U, torch::Tensor indexes, torch::Tensor nvals, torch::Tensor W, torch::Tensor C);

torch::Tensor get_wmma_uw_contraction_fp32(torch::Tensor U, torch::Tensor indexes, torch::Tensor nvals, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	WMMATensor3Contraction_fp32(U, indexes, nvals, W, output);

	return output;
}

torch::Tensor get_sparse_uw_contraction_fp32(torch::Tensor nnon_zero, torch::Tensor indices, torch::Tensor U, torch::Tensor W) {
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	SparseTensor3Contraction_fp32(nnon_zero, indices, U, W, output);

	return output;
}

torch::Tensor get_sparse_uw_contraction_fp64(torch::Tensor nnon_zero, torch::Tensor indices, torch::Tensor U, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat64).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	SparseTensor3Contraction_fp64(nnon_zero, indices, U, W, output);

	return output;

}

torch::Tensor get_uw_contraction_fp32(torch::Tensor U, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	DenseTensor3Contraction_fp32(U, W, output);

	return output;

}

torch::Tensor get_uw_contraction_fp64(torch::Tensor U, torch::Tensor W) {

	auto options = torch::TensorOptions().dtype(torch::kFloat64).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { U.size(0), U.size(1), U.size(2), W.size(1) }, options);

	DenseTensor3Contraction_fp64(U, W, output);

	return output;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_uw_contraction_fp32", &get_uw_contraction_fp32, "");
	m.def("get_uw_contraction_fp64", &get_uw_contraction_fp64, "");
	m.def("get_sparse_uw_contraction_fp32", &get_sparse_uw_contraction_fp32, "");
	m.def("get_sparse_uw_contraction_fp64", &get_sparse_uw_contraction_fp64, "");

	m.def("get_wmma_uw_contraction_fp32", &get_wmma_uw_contraction_fp32, "");

}
