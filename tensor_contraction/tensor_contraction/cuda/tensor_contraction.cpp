#include <torch/extension.h>
#include<iostream>

using namespace std;

void U3W3_X_contraction_gpu(
							torch::Tensor UW, 
							torch::Tensor UW_nonsparse_indices, 
							torch::Tensor UW_nonsparse_num_nonzeros,
							torch::Tensor X,
							torch::Tensor atom_types,
							torch::Tensor out
							);


torch::Tensor U3W3_X_contraction(
							torch::Tensor UW, 
							torch::Tensor UW_nonsparse_indices, 
							torch::Tensor UW_nonsparse_num_nonzeros,
							torch::Tensor X,
							torch::Tensor atom_types
							) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { X.size(0), UW.size(0), UW.size(1), UW.size(4)}, options);

	U3W3_X_contraction_gpu(UW, UW_nonsparse_indices, UW_nonsparse_num_nonzeros, X, atom_types, output);

	return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("U3W3_X_contraction", &U3W3_X_contraction, "");
}
