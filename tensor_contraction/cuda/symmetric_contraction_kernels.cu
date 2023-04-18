#include <math.h>
#include <torch/torch.h>
#include <iostream>

using namespace std;


// (nl + 1) ** 2, (nl + 1) ** 2, n_nonzero
 // (nl + 1) ** 2, (nl + 1) ** 2
  // (nl + 1) ** 2, (nl + 1) ** 2, (nl + 1) ** 2, nelements, nchannels
// natoms
// natoms, (nl + 1) ** 2, nchannels
 // natoms, (nl + 1) ** 2, (nl + 1) ** 2, nchannels ?
 

__global__ void sparse_symmetric_contraction_derivative_kernel(
	const torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> UW3_nonsparse_indices,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> UW3_num_nonsparse,
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW3,
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW3_deriv_factors,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> UW2,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> UW1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_out)
{

	extern __shared__ char buffer[];

	const int natoms = X.size(0);
	const int nl = X.size(1);
	const int nchannels = X.size(2);

	size_t offset = 0;

	float *buffer_X = reinterpret_cast<float *>(buffer + offset);
	offset +=  blockDim.x * nl * sizeof(float);
	float *buffer_grad = reinterpret_cast<float *>(buffer + offset);
	offset +=  blockDim.x * nl * sizeof(float);
	float *buffer_out = reinterpret_cast<float *>(buffer + offset);
	offset +=  blockDim.x * sizeof(float);

	uint8_t * buffer_uw3_num_nonsparse =  reinterpret_cast<uint8_t *>(buffer + offset);
	offset += nl * nl  * sizeof(uint8_t);
	uint8_t * buffer_uw3_nonsparse_indices =  reinterpret_cast<uint8_t *>(buffer + offset);
	offset +=  UW3_num_nonsparse.size(2) * nl * nl  * sizeof(uint8_t);

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int n_threads_x = blockDim.x;
	int n_threads_y = blockDim.y;
	int n_threads_z = blockDim.z;

	int n_blocks_x = gridDim.x;
	int n_blocks_y = gridDim.y;
	int n_blocks_z = gridDim.z;

	if (threadIdx.z == 0) {
		for (int i = threadIdx.y; i < nl; i += blockDim.y) {
			for (int j = threadIdx.x; j < nl; j += blockDim.x) {

				int nsparse = UW3_num_nonsparse[i][j];

				buffer_uw3_num_nonsparse[i * nl + j] = nsparse;

				// 16, 16, 3
				for (int k = 0; k < nsparse; k ++) {
					buffer_uw3_nonsparse_indices[i * (nl * 3) + (k * nl) + j] = UW3_nonsparse_indices[i][j][k]; //todo account for k here
				}
			} 
		}
	}

	__syncthreads();

	for (int atom_id = blockIdx.x; atom_id < natoms; atom_id +=gridDim.x) {

		int element = atom_types[atom_id];

		for (int channel_id = threadIdx.x; channel_id < X.size(2); channel_id += blockDim.x){
			
			for (int i = threadIdx.y; i <nl; i += blockDim.y) {

				float Xir = X[atom_id][i][channel_id];

				buffer_X[i * blockDim.x + threadIdx.x] =  Xir;
				buffer_grad[i * blockDim.x + threadIdx.x] = 0.0;
			}

			buffer_out[threadIdx.x] = 0.0;
			
			__syncthreads();

			float output_1 = 0.0;

			for (int i = threadIdx.y; i < nl; i += blockDim.y) {
				
				float Xi = buffer_X[i * blockDim.x + threadIdx.x]; //X[atom_idx][j][r];

				float uw1_i =  UW1[i][element][channel_id];

				float deriv1_tmp = uw1_i;

				float output_2 = 0.0;

				for (int j = threadIdx.z; j < nl; j +=blockDim.z) {
									
					uint8_t uw3_num_nonsparse = buffer_uw3_num_nonsparse[i * nl + j];

					float Xj = buffer_X[j * blockDim.x + threadIdx.x]; //X[atom_idx][j][r];

					float uw2_ij  = UW2[i][j][element][channel_id]; // UW2 is symmetric in [i, j]

					//float deriv_1_j_tmp = uw2_ij ;
					float deriv_1_j_tmp = uw2_ij ;

					float output_3 = 0.0;

					for (uint8_t k = 0; k < uw3_num_nonsparse; k ++) {

						uint8_t kdx = buffer_uw3_nonsparse_indices[i * (nl * 3) + (k * nl) + j];
						
						float Xk = buffer_X[kdx * blockDim.x + threadIdx.x];

						//float uw3_jki = UW3[j][kdx][i][element][channel_id];
						float uw3_ijk = UW3[i][j][kdx][element][channel_id]; 
						//float uw3_jik = UW3[j][i][kdx][element][channel_id];

						float factor = UW3_deriv_factors[i][j][kdx][element][channel_id]; 

						output_3 += uw3_ijk * Xk;

						deriv_1_j_tmp += factor * Xk;
					}

					output_2 += (output_3 + uw2_ij) * Xj;

					deriv1_tmp +=  (uw2_ij + deriv_1_j_tmp) * Xj;
				}

				output_1 += (output_2 + uw1_i) * Xi;

				grad_out[atom_id][i][channel_id] = deriv1_tmp;
				//atomicAdd(&buffer_grad[i * blockDim.x + threadIdx.x], deriv1_tmp);
			}
			
			atomicAdd(&buffer_out[threadIdx.x], output_1); 

			__syncthreads();

			//for (int i = threadIdx.y; i < nl; i +=blockDim.y) {
			//	grad_out[atom_id][i][channel_id] = buffer_grad[i * blockDim.x + threadIdx.x];
			//}

			if (threadIdx.y == 0) {
				out[atom_id][channel_id] = buffer_out[threadIdx.x];
			}
		}
	}
}


std::vector<torch::Tensor> sparse_symmetric_contraction_derivative(
	torch::Tensor UW3_nonsparse_indices,
	torch::Tensor UW3_num_nonsparse,
	torch::Tensor UW3,
	torch::Tensor UW3_deriv_factors,
	torch::Tensor UW2,
	torch::Tensor UW1,
	torch::Tensor X,
	torch::Tensor atom_types)
{
	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(X.dtype()).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor out = torch::empty({X.size(0), X.size(2)}, options);

	torch::Tensor grad;

	if (X.requires_grad()) {
		grad = torch::empty_like(X, options);
	} else {
		grad = torch::empty({1, 1, 1}, options);
	}

	dim3 block_dim(X.size(0));

	dim3 grid(16, 16, 1);

	AT_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "sparse_symmetric_contraction_derivative_kernel", ([&] {

			size_t shared_mem_amount =  nthreadX * X.size(1) * sizeof(scalar_t); // X storage
			shared_mem_amount +=  nthreadX * X.size(1) * sizeof(scalar_t); // grad stoage
			shared_mem_amount +=  nthreadX * sizeof(scalar_t); // output stoage
			shared_mem_amount += X.size(1) * X.size(1) * sizeof(uint8_t); // uw3_num_nonsparse stoage
			shared_mem_amount += 3 * X.size(1) * X.size(1) * sizeof(uint8_t); // uw3_nonsparse_indices storage
           
			sparse_symmetric_contraction_derivative_kernel<<<block_dim, grid, shared_mem_amount>>>(
			UW3_nonsparse_indices.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
			UW3_num_nonsparse.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
			UW3.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
			UW3_deriv_factors.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
			UW2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			UW1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
			X.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
			atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			grad_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());

        }));

	
	cudaDeviceSynchronize();

	return {out, grad};
}

torch::Tensor backwards(torch::Tensor symm_contraction_grad, torch::Tensor grad_outputs) {

	AT_DISPATCH_FLOATING_TYPES(
        symm_contraction_grad.scalar_type(), "backwards_kernel", ([&] {

	
	}));
}


class SymmetricContractionAutograd : public torch::autograd::Function<SymmetricContractionAutograd> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
		torch::Tensor X,
		torch::Tensor atom_types,
        torch::Tensor UW3_nonsparse_indices,
		torch::Tensor UW3_num_nonsparse,
		torch::Tensor UW3,
		torch::Tensor UW3_deriv_factors,
		torch::Tensor UW2,
		torch::Tensor UW1,
    ) {

		if (!X.is_contiguous()) {
        	throw std::runtime_error("this code only runs with contiguous tensors");
    	}

		if (!X.device().is_cpu()) {
			throw std::runtime_error("internal error: called CPU version on non-CPU tensor");
		}

		auto result = sparse_symmetric_contraction_derivative(UW3_nonsparse_indices, UW3_num_nonsparse, UW3, UW3_deriv_factors, UW2, UW1, X, atom_types);

		auto symm_contraction = result[0];
		auto symm_contraction_grad = result[1];

		if (xyz.requires_grad()) {
        	ctx->save_for_backward({symm_contraction_grad});
		}

		return symm_contraction;
    }
	

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {

		 auto saved_variables = ctx->get_saved_variables();

		auto symm_contraction_grad = saved_variables[0];	
    	auto grad_outputs = grad_outputs[0];

		auto grad = backwards_cuda(symm_contraction_grad, grad_outputs);

		Torch::Tensor nothing = torch::Tensor();

		return {grad, nothing, nothing, nothing, nothing, nothing, nothing, nothing};

	}
};

Tensor symmetric_contraction(       
		torch::Tensor X,
		torch::Tensor atom_types,
	    torch::Tensor UW3_nonsparse_indices,
		torch::Tensor UW3_num_nonsparse,
		torch::Tensor UW3,
		torch::Tensor UW3_deriv_factors,
		torch::Tensor UW2,
		torch::Tensor UW1
) {
  return SymmetricContractionAutograd::apply(X, atom_types, UW3_nonsparse_indices, UW3_num_nonsparse, UW3,UW3_deriv_factors, UW2, UW1);
}

TORCH_LIBRARY(mace_ops, m) {
  m.def("symmetric_contraction(
	Tensor X, 
	Tensor atom_types, 
	Tensor UW3_nonsparse_indices, 
	Tensor UW3_num_nonsparse, 
  	Tensor UW3, 
	Tensor UW3_deriv_factors, 
	Tensor UW2, 
	Tensor UW1) -> Tensor", symmetric_contraction);
}

