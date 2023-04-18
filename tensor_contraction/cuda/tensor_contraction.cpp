#include <torch/extension.h>
#include <iostream>

using namespace std;

void sparse_full_symmetric_contraction_gpu(
	torch::Tensor U3_nonsparse_indices,
	torch::Tensor U3_num_nonsparse,
	torch::Tensor U3_nonsparse_elements,
	torch::Tensor U2,
	torch::Tensor U1,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1); 

/*void sparse_full_symmetric_contraction_derivative_gpu(
	torch::Tensor U3_nonsparse_indices,
	torch::Tensor U3_num_nonsparse,
	torch::Tensor U3_nonsparse_elements,
	torch::Tensor U2,
	torch::Tensor U1,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	torch::Tensor grad_out,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1); */



void sparse_symmetric_contraction_derivative_gpu(
	torch::Tensor UW3_nonsparse_indices,
	torch::Tensor UW3_num_nonsparse,
	torch::Tensor UW3,
	torch::Tensor UW3_deriv_factors,
	torch::Tensor UW2,
	torch::Tensor UW1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	torch::Tensor grad_out,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1);

void sparse_symmetric_contraction_gpu(
	torch::Tensor UW3_nonsparse_indices,
	torch::Tensor UW3_num_nonsparse,
	torch::Tensor UW3,
	torch::Tensor UW2,
	torch::Tensor UW1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1);



torch::Tensor sparse_symmetric_contraction(
	torch::Tensor UW3_nonsparse_indices,
	torch::Tensor UW3_num_nonsparse,
	torch::Tensor UW3,
	torch::Tensor UW2,
	torch::Tensor UW1,
	torch::Tensor X,
	torch::Tensor atom_types,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1)
{
	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor out = torch::empty({X.size(0), X.size(2)}, options);

	sparse_symmetric_contraction_gpu(UW3_nonsparse_indices, UW3_num_nonsparse, UW3, UW2, UW1,
									X, atom_types, out, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);


	return out;
}

std::vector<torch::Tensor> sparse_symmetric_contraction_derivative(
	torch::Tensor UW3_nonsparse_indices,
	torch::Tensor UW3_num_nonsparse,
	torch::Tensor UW3,
	torch::Tensor UW3_deriv_factors,
	torch::Tensor UW2,
	torch::Tensor UW1,
	torch::Tensor X,
	torch::Tensor atom_types,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 16,
	int nthreadY = 16,
	int nthreadZ = 1)
{
	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor grad = torch::empty_like(X, options);
	torch::Tensor out = torch::empty({X.size(0), X.size(2)}, options);

	sparse_symmetric_contraction_derivative_gpu(UW3_nonsparse_indices, UW3_num_nonsparse, UW3, UW3_deriv_factors, UW2, UW1,
									X, atom_types, out, grad, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);


	return {out, grad};
}

std::vector<torch::Tensor> sparse_full_symmetric_contraction(
	torch::Tensor U3_nonsparse_indices,
	torch::Tensor U3_num_nonsparse,
	torch::Tensor U3_nonsparse_elements,
	torch::Tensor U2,
	torch::Tensor U1,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1)
{
	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor out = torch::empty({X.size(0), X.size(2)}, options);

	sparse_full_symmetric_contraction_gpu(U3_nonsparse_indices, U3_num_nonsparse, U3_nonsparse_elements, U2, U1,
									W3, W2, W1, X, atom_types, out, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);


	return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("sparse_symmetric_contraction_derivative", &sparse_symmetric_contraction_derivative, "");
	m.def("sparse_symmetric_contraction", &sparse_symmetric_contraction, "");
	m.def("sparse_full_symmetric_contraction", &sparse_full_symmetric_contraction, "");
}
