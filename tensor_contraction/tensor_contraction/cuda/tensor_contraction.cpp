#include <torch/extension.h>
#include<iostream>

using namespace std;

void correlation_3_main_gpu(
							torch::Tensor UW, 
							torch::Tensor UW_nonsparse_indices, 
							torch::Tensor UW_nonsparse_num_nonzeros,
							torch::Tensor X,
							torch::Tensor atom_types,
							torch::Tensor out,
							int nblockX=1,
							int nblockY=16,
							int nblockZ=1,
							int nthreadX=32,
							int nthreadY=16,
							int nthreadZ=1
							);


void correlation_2_contraction_gpu(
							torch::Tensor U2_nonsparse_indices, 
							torch::Tensor element_weights,
							torch::Tensor input_weights,
							torch::Tensor X,
							torch::Tensor atom_types,
							torch::Tensor out,
							int nblockX=1,
							int nblockY=1,
							int nblockZ=1,
							int nthreadX=32,
							int nthreadY=16,
							int nthreadZ=1
							);

void correlation_1_contraction_gpu(
							torch::Tensor U1_nonsparse_indices, 
							torch::Tensor element_weights,
							torch::Tensor input_weights,
							torch::Tensor X,
							torch::Tensor atom_types,
							torch::Tensor out,
							int nblockX=1,
							int nblockY=1,
							int nblockZ=1,
							int nthreadX=32,
							int nthreadY=1,
							int nthreadZ=1
							);



torch::Tensor correlation_3_main(
							torch::Tensor UW, 
							torch::Tensor UW_nonsparse_indices, 
							torch::Tensor UW_nonsparse_num_nonzeros,
							torch::Tensor X,
							torch::Tensor atom_types,
							int nblockX=1,
							int nblockY=16,
							int nblockZ=1,
							int nthreadX=32,
							int nthreadY=16,
							int nthreadZ=1
							) {

	int bx = nblockX;

	if (bx == 1){
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { X.size(0), UW.size(0), UW.size(1), UW.size(4)}, options);

	correlation_3_main_gpu(UW, UW_nonsparse_indices, UW_nonsparse_num_nonzeros, 
							X, atom_types, output, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);

	return output;
}

torch::Tensor correlation_2_contraction(
							torch::Tensor U2_nonsparse_indices, 
							torch::Tensor element_weights,
							torch::Tensor input_weights,
							torch::Tensor X,
							torch::Tensor atom_types,
							int nblockX=1,
							int nblockY=1,
							int nblockZ=1,
							int nthreadX=32,
							int nthreadY=16,
							int nthreadZ=1
							) {

	int bx = nblockX;

	if (bx == 1){
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { X.size(0), X.size(1), X.size(2)}, options);

	correlation_2_contraction_gpu(U2_nonsparse_indices, element_weights, input_weights,
							X, atom_types, output, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);

	return output;
}

torch::Tensor correlation_1_contraction(
							torch::Tensor U1_nonsparse_indices, 
							torch::Tensor element_weights,
							torch::Tensor input_weights,
							torch::Tensor X,
							torch::Tensor atom_types,
							int nblockX=1,
							int nblockY=1,
							int nblockZ=1,
							int nthreadX=32,
							int nthreadY=1,
							int nthreadZ=1
							) {

	int bx = nblockX;

	if (bx == 1){
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(
			torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { X.size(0), X.size(2)}, options);

	correlation_1_contraction_gpu(U1_nonsparse_indices, element_weights, input_weights, X,
						   atom_types, output, bx, nblockY, nblockZ, nthreadX, nthreadY, nthreadZ);

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

	m.def("correlation_3_main", &correlation_3_main, "");
	m.def("correlation_2_contraction", &correlation_2_contraction, "");
	m.def("correlation_1_contraction", &correlation_1_contraction, "");
}
