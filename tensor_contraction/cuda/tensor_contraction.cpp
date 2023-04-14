#include <torch/extension.h>
#include <iostream>

using namespace std;

void correlation_3_main_gpu(
	torch::Tensor UW,
	torch::Tensor UW_nonsparse_indices,
	torch::Tensor UW_nonsparse_num_nonzeros,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	torch::Tensor grad,
	bool requires_grad,
	int nblockX = 1,
	int nblockY = 16,
	int nblockZ = 1,
	int nthreadX = 32,
	int nthreadY = 16,
	int nthreadZ = 1);
void correlation_3_main_and_grad_gpu(
	torch::Tensor UW,
	torch::Tensor UW_nonsparse_indices,
	torch::Tensor UW_nonsparse_num_nonzeros,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	torch::Tensor grad,
	bool requires_grad,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1);

void correlation_2_contraction_gpu(
	torch::Tensor UW2,
	torch::Tensor prev_layer,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor grad_in,
	torch::Tensor out,
	torch::Tensor grad_out,
	bool requires_grad,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 32,
	int nthreadY = 16,
	int nthreadZ = 1);

void correlation_1_contraction_gpu(
	torch::Tensor UW1,
	torch::Tensor prev_layer,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor grad_in,
	torch::Tensor out,
	torch::Tensor grad_out,
	bool requires_grad,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 32,
	int nthreadY = 1,
	int nthreadZ = 1);

void symmetric_contraction_derivative_gpu(
	torch::Tensor UW3,
	torch::Tensor UW2,
	torch::Tensor UW1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor grad_out,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1);


void sparse_symmetric_contraction_derivative_gpu(
	torch::Tensor UW3_nonsparse_indices,
	torch::Tensor UW3_num_nonsparse,
	torch::Tensor UW3,
	torch::Tensor UW2,
	torch::Tensor UW1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor grad_out,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1);


torch::Tensor symmetric_contraction_derivative(
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

	torch::Tensor grad = torch::empty_like(X, options);
	
	symmetric_contraction_derivative_gpu(UW3, UW2, UW1,
									X, atom_types,  grad, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);


	return grad;
}


torch::Tensor sparse_symmetric_contraction_derivative(
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

	torch::Tensor grad = torch::empty_like(X, options);
	
	sparse_symmetric_contraction_derivative_gpu(UW3_nonsparse_indices, UW3_num_nonsparse, UW3, UW2, UW1,
									X, atom_types,  grad, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);


	return grad;
}

std::vector<torch::Tensor> correlation_3_main_and_grad(
	torch::Tensor UW,
	torch::Tensor UW_nonsparse_indices,
	torch::Tensor UW_nonsparse_num_nonzeros,
	torch::Tensor X,
	torch::Tensor atom_types,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 16,
	int nthreadY = 2,
	int nthreadZ = 8)
{

	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros({X.size(0), UW.size(0), UW.size(1), UW.size(4)}, options);

	torch::Tensor grad;

	if (X.requires_grad())
	{
		grad = torch::zeros({X.size(0), X.size(1), X.size(2)}, options);
	}
	else
	{
		grad = torch::zeros({1, 1, 1}, options);
	}

	correlation_3_main_and_grad_gpu(UW, UW_nonsparse_indices, UW_nonsparse_num_nonzeros,
									X, atom_types, output, grad, X.requires_grad(), bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);

	if (X.requires_grad())
	{
		return {output, grad};
	}
	else
	{
		return {output};
	}
}

std::vector<torch::Tensor> correlation_3_main(
	torch::Tensor UW,
	torch::Tensor UW_nonsparse_indices,
	torch::Tensor UW_nonsparse_num_nonzeros,
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

	torch::Tensor output = torch::zeros({X.size(0), UW.size(0), UW.size(1), UW.size(4)}, options);

	torch::Tensor grad_out;

	if (X.requires_grad())
	{
		grad_out = torch::zeros({X.size(0), X.size(1), X.size(2)}, options);
	}
	else
	{
		grad_out = torch::zeros({1, 1, 1}, options);
	}

	correlation_3_main_gpu(UW, UW_nonsparse_indices, UW_nonsparse_num_nonzeros,
						   X, atom_types, output, grad_out, X.requires_grad(), bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);


	return {output, grad_out};
	
}

std::vector<torch::Tensor>  correlation_2_contraction(
	torch::Tensor UW2,
	torch::Tensor prev_layer,
	torch::Tensor X,
	torch::Tensor grad_in,
	torch::Tensor atom_types,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 32,
	int nthreadY = 16,
	int nthreadZ = 1)
{

	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros({X.size(0), X.size(1), X.size(2)}, options);

	torch::Tensor grad_out;

	if (X.requires_grad())
	{
		grad_out = torch::zeros({X.size(0), X.size(1), X.size(2)}, options);
	}
	else
	{
		grad_out = torch::zeros({1, 1, 1}, options);
	}

	correlation_2_contraction_gpu(UW2, prev_layer,
								  X, atom_types, grad_in, output, grad_out, X.requires_grad(), bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);

	return {output, grad_out};
}

std::vector<torch::Tensor>  correlation_1_contraction(
	torch::Tensor UW1,
	torch::Tensor prev_layer,
	torch::Tensor X,
	torch::Tensor grad_in,
	torch::Tensor atom_types,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 32,
	int nthreadY = 1,
	int nthreadZ = 1)
{

	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros({X.size(0), X.size(2)}, options);

	torch::Tensor grad_out;

	if (X.requires_grad())
	{
		grad_out = torch::zeros({X.size(0), X.size(1), X.size(2)}, options);
	}
	else
	{
		grad_out = torch::zeros({1, 1, 1}, options);
	}

	correlation_1_contraction_gpu(UW1, prev_layer, X,
								  atom_types, grad_in, output, grad_out, X.requires_grad(), bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);

	return {output, grad_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

	m.def("correlation_3_main", &correlation_3_main, "");
	m.def("correlation_3_main_and_grad", &correlation_3_main_and_grad, "");
	m.def("correlation_2_contraction", &correlation_2_contraction, "");
	m.def("correlation_1_contraction", &correlation_1_contraction, "");
	m.def("symmetric_contraction_derivative", &symmetric_contraction_derivative, "");
	
	m.def("sparse_symmetric_contraction_derivative", &sparse_symmetric_contraction_derivative, "");

}
