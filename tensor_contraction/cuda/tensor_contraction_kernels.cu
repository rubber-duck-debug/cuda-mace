#include <math.h>
#include<torch/torch.h>
#include <iostream>

using namespace std;

__global__ void correlation_3_main_kernel(
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW,
	const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> UW_nonsparse_indices,
	const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> UW_num_nonzeros,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X, 
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types, 
	torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> out){

	int n_iter_x = X.size(2) / n_threads_x;
	int nblock_iter_x = X.size(0) / n_blocks_x;
	int nblock_iter_y = UW.size(0) / n_blocks_y;
	int nblock_iter_z = UW.size(1) / n_blocks_z;

	for (int gy = by; gy < UW.size(0); gy += n_blocks_y) {

		int num_nonsparse = UW_num_nonzeros[gy][ty];
		
		for (int atom_idx = bx; atom_idx < X.size(0); atom_idx +=n_blocks_x){

			int element = atom_types[atom_idx];

			for (int l = 0; l < num_nonsparse; l++ ) {

				int ldx = UW_nonsparse_indices[gy][ty][l];

				for (int i = 0; i < n_iter_x; i ++){

					int idx_i = i * n_threads_x + tx;

					float uw =  UW[gy][ty][ldx][element][idx_i];
					float x = X[atom_idx][ldx][idx_i];

					atomicAdd(&out[atom_idx][gy][ty][idx_i], uw * x);
				}
			}
		}
	}
}

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
							) {

	
	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	correlation_3_main_kernel<<<blocks, grid>>>(
			UW.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
			UW_nonsparse_indices.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			UW_nonsparse_num_nonzeros.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), 
			atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), 
			out.packed_accessor32<float, 4, torch::RestrictPtrTraits>()
			);

	cudaDeviceSynchronize();

}

__global__ void correlation_2_contraction_kernel(
	const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> U2_nonsparse_indices,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> element_weights,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> input_weights,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X, 
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out){


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

	int n_iter_x = X.size(2) / n_threads_x;
	int n_iter_y = U2_nonsparse_indices.size(0) / n_threads_y;

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx +=n_blocks_x){
		
		int element = atom_types[atom_idx];

		for (int gy = ty; gy < U2_nonsparse_indices.size(0); gy += n_threads_y) {

			int ix = U2_nonsparse_indices[gy][0];
			int jx = U2_nonsparse_indices[gy][1];
			int kx = U2_nonsparse_indices[gy][2];

				for (int i = 0; i < n_iter_x; i ++){

					int idx_i = i * n_threads_x + tx;

					float w =  input_weights[atom_idx][ix][jx][idx_i] + element_weights[element][ix][jx][idx_i];

					float x = X[atom_idx][jx][idx_i];

					atomicAdd(&out[atom_idx][ix][idx_i], w * x);
				}
			
		}
	}
}

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
							) {

	
	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	correlation_2_contraction_kernel<<<blocks, grid>>>(
			U2_nonsparse_indices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			element_weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			input_weights.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), 
			atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), 
			out.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
			);

	cudaDeviceSynchronize();

}

__global__ void correlation_1_contraction_kernel(
	const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> U1_nonsparse_indices,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> element_weights,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> input_weights,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X, 
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out){


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

	int n_iter_x = X.size(2) / n_threads_x;

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx +=n_blocks_x){
		
		int element = atom_types[atom_idx];

		for (int gy = ty; gy < U1_nonsparse_indices.size(0); gy += n_threads_y) {

			int ix = U1_nonsparse_indices[gy][0];
			int jx = U1_nonsparse_indices[gy][1];

				for (int i = 0; i < n_iter_x; i ++){

					int idx_i = i * n_threads_x + tx;

					float w =  input_weights[atom_idx][jx][idx_i] + element_weights[element][jx][idx_i];

					float x = X[atom_idx][ix][idx_i];

					atomicAdd(&out[atom_idx][idx_i], w * x);
				}
			
		}
	}
}

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
							) {

	
	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	correlation_1_contraction_kernel<<<blocks, grid>>>(
			U1_nonsparse_indices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			element_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			input_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), 
			atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(), 
			out.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
			);

	cudaDeviceSynchronize();

}
