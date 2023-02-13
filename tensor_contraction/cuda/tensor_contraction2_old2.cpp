#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void u4w_matmul_tc16x16_f32(torch::Tensor U, torch::Tensor W, torch::Tensor C);

torch::Tensor get_u4w_matmul_tc16x16_f32(torch::Tensor A, torch::Tensor B) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { A.size(0), A.size(1), B.size(1) }, options);

	u4w_matmul_tc16x16_f32(A, B, output);

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

	m.def("get_u4w_matmul_tc16x16_f32", &get_u4w_matmul_tc16x16_f32, "");

}
