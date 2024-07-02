#ifndef SYMMETRIC_CONTRACITON_H
#define SYMMETRIC_CONTRACITON_H

#include <torch/script.h>
#include <cooperative_groups.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;


class SymmetricContractionAutograd : public Function<SymmetricContractionAutograd>
{
public:
	static torch::Tensor forward(
		AutogradContext *ctx,
		torch::Tensor X,
		torch::Tensor atom_types,
		torch::Tensor U3_num_nonzero_1,
		torch::Tensor U3_num_nonzero_2,
		torch::Tensor U3_num_nonzero_3,
		torch::Tensor U3_indices_0,
		torch::Tensor U3_indices_1,
		torch::Tensor U3_indices_2,
		torch::Tensor U3_indices_3,
		torch::Tensor U3_values_0,
		torch::Tensor U3_values_1,
		torch::Tensor U3_values_2,
		torch::Tensor U3_values_3,
		torch::Tensor U2_num_nonzero_1,
		torch::Tensor U2_num_nonzero_2,
		torch::Tensor U2_indices_1,
		torch::Tensor U2_indices_2,
		torch::Tensor U2_values_1,
		torch::Tensor U2_values_2,
		torch::Tensor U1_num_nonzero,
		torch::Tensor U1_index,
		torch::Tensor W3,
		torch::Tensor W2,
		torch::Tensor W1,
		const int64_t W3_L0_size,
		const int64_t W2_L0_size,
		const int64_t W1_L0_size,
		torch::Tensor W3_size,			// nlout
		torch::Tensor W2_size,			// nlout
		torch::Tensor W1_size,			// nlout
		torch::Tensor U3_max_nonsparse, // nlout
		const int64_t nthreadx,
		const int64_t nthready,
		const int64_t nthreadz);

	static variable_list backward(AutogradContext *ctx, variable_list grad_outputs);
};

class SymmetricContraction : public torch::CustomClassHolder
{
public:
	torch::Tensor forward(
		torch::Tensor X,
		torch::Tensor atom_types,
		torch::Tensor U3_num_nonzero_1,
		torch::Tensor U3_num_nonzero_2,
		torch::Tensor U3_num_nonzero_3,
		torch::Tensor U3_indices_0,
		torch::Tensor U3_indices_1,
		torch::Tensor U3_indices_2,
		torch::Tensor U3_indices_3,
		torch::Tensor U3_values_0,
		torch::Tensor U3_values_1,
		torch::Tensor U3_values_2,
		torch::Tensor U3_values_3,
		torch::Tensor U2_num_nonzero_1,
		torch::Tensor U2_num_nonzero_2,
		torch::Tensor U2_indices_1,
		torch::Tensor U2_indices_2,
		torch::Tensor U2_values_1,
		torch::Tensor U2_values_2,
		torch::Tensor U1_num_nonzero,
		torch::Tensor U1_index,
		torch::Tensor W3,
		torch::Tensor W2,
		torch::Tensor W1,
		const int64_t W3_L0_size,
		const int64_t W2_L0_size,
		const int64_t W1_L0_size,
		torch::Tensor W3_size,			// nlout
		torch::Tensor W2_size,			// nlout
		torch::Tensor W1_size,			// nlout
		torch::Tensor U3_max_nonsparse, // nlout
		const int64_t nthreadx,
		const int64_t nthready,
		const int64_t nthreadz);

	std::vector<torch::Tensor> __getstate__()
	{
		return {};
	}

	void __setstate__(const std::vector<torch::Tensor> &state)
	{
		return;
	}
};

#endif
