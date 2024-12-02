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
		const int64_t U3_max_nonsparse,
		torch::Tensor U3_num_nonzero,
		torch::Tensor U3_indices,
		torch::Tensor U3_values,
		torch::Tensor U2_num_nonzero,
		torch::Tensor U2_indices,
		torch::Tensor U2_values,
		torch::Tensor U1_num_nonzero,
		torch::Tensor U1_index,
		torch::Tensor W3,
		torch::Tensor W2,
		torch::Tensor W1,
		const int64_t W3_size,
		const int64_t W2_size,
		const int64_t W1_size
		);

	static variable_list backward(AutogradContext *ctx, variable_list grad_outputs);
};

class SymmetricContraction : public torch::CustomClassHolder
{
public:
	torch::Tensor forward(
		torch::Tensor X,
		torch::Tensor atom_types,
		const int64_t U3_max_nonsparse,
		torch::Tensor U3_num_nonzero,
		torch::Tensor U3_indices,
		torch::Tensor U3_values,
		torch::Tensor U2_num_nonzero,
		torch::Tensor U2_indices,
		torch::Tensor U2_values,
		torch::Tensor U1_num_nonzero,
		torch::Tensor U1_index,
		torch::Tensor W3,
		torch::Tensor W2,
		torch::Tensor W1,
		const int64_t W3_size,
		const int64_t W2_size,
		const int64_t W1_size
		);

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
