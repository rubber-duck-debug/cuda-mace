#include <math.h>
#include <torch/torch.h>
#include <iostream>

using namespace std;

#define FULL_MASK 0xffffffff


__global__ void sparse_symmetric_contraction_kernel(
	const torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> UW3_nonsparse_indices,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> UW3_num_nonsparse,
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW3,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> UW2,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> UW1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
{

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_X = reinterpret_cast<float *>(buffer + offset);
	offset += blockDim.z * blockDim.x * X.size(1) * sizeof(float);
	float *buffer_out = reinterpret_cast<float *>(buffer + offset);
	offset += blockDim.z * blockDim.x * sizeof(float);

	uint8_t * buffer_uw3_num_nonsparse =  reinterpret_cast<uint8_t *>(buffer + offset);
	offset +=  X.size(1) * X.size(1)  * sizeof(uint8_t);
	uint8_t * buffer_uw3_nonsparse_indices =  reinterpret_cast<uint8_t *>(buffer + offset);
	offset +=  3 * X.size(1) * X.size(1)  * sizeof(uint8_t);

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

	int atom_idx = blockIdx.x * blockDim.z + threadIdx.z;

	if (threadIdx.z == 0) {
		for (int i = threadIdx.y; i < X.size(1); i += blockDim.y) {
			for (int j = threadIdx.x; j < X.size(1); j += blockDim.x) {

				int nsparse = UW3_num_nonsparse[i][j];

				buffer_uw3_num_nonsparse[i * X.size(1) + j] = nsparse;

				// 16, 16, 3
				for (int k = 0; k < nsparse; k ++) {
					buffer_uw3_nonsparse_indices[i * (X.size(1) * 3) + (k * X.size(1)) + j] = UW3_nonsparse_indices[i][j][k]; //todo account for k here
				}
			} 
		}
	}

	__syncthreads();

	int element = 0;

	if (atom_idx < X.size(0)){
		element = atom_types[atom_idx];
	}

	for (int r = threadIdx.x; r < X.size(2); r+= blockDim.x){
		
		for (int i = threadIdx.y; i < X.size(1); i += blockDim.y) {

			float Xir = 0.0;

			if (atom_idx < X.size(0)){
				 Xir = X[atom_idx][i][r];
			}

			buffer_X[threadIdx.z * (X.size(1) * blockDim.x) + i * blockDim.x + threadIdx.x] =  Xir;
		}

		buffer_out[threadIdx.z * blockDim.x + threadIdx.x] = 0.0;
		
		__syncthreads();

		float output_1 = 0.0;

		for (int i = threadIdx.y; i < X.size(1); i += blockDim.y) {
			
			float Xi = buffer_X[threadIdx.z * (X.size(1) * blockDim.x) + i * blockDim.x + threadIdx.x]; //X[atom_idx][j][r];

			float uw1_i =  UW1[i][element][r];

			float deriv1_tmp = uw1_i;

			float output_2 = 0.0;

			for (int j = 0; j < X.size(1); j ++) {
								
				uint8_t uw3_num_nonsparse = buffer_uw3_num_nonsparse[i * X.size(1) + j];

				float Xj = buffer_X[threadIdx.z * (X.size(1) * blockDim.x) + j * blockDim.x + threadIdx.x]; //X[atom_idx][j][r];

				float uw2_ij  = UW2[i][j][element][r]; // UW2 is symmetric in [i, j]

				float output_3 = 0.0;

				for (uint8_t k = 0; k < uw3_num_nonsparse; k ++) {

					uint8_t kdx1 = buffer_uw3_nonsparse_indices[i * (X.size(1) * 3) + (k * X.size(1)) + j];

					float Xk1 = buffer_X[threadIdx.z * (X.size(1) * blockDim.x) + kdx1 * blockDim.x + threadIdx.x];

					float uw3_ijk = UW3[i][j][kdx1][element][r]; 

					output_3 += uw3_ijk * Xk1;
				}

				output_2 += (output_3 + uw2_ij) * Xj;

			}

			output_1 += (output_2 + uw1_i) * Xi;

		}
		
		atomicAdd(&buffer_out[threadIdx.z * blockDim.x + threadIdx.x], output_1); 

		__syncthreads();


		if (atom_idx < X.size(0)) {

			if (threadIdx.y == 0) {
				out[atom_idx][r] = buffer_out[threadIdx.z * blockDim.x + threadIdx.x];
			}
		}
	}
}


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
	int nthreadZ = 1)
{

	auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    dim3 block_dim(find_num_blocks(X.size(0), nthreadZ));

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t shared_mem_amount = nthreadZ * nthreadX * X.size(1) * sizeof(float); // X storage
	shared_mem_amount += nthreadZ * nthreadX * sizeof(float); // output stoage
	shared_mem_amount += X.size(1) * X.size(1) * sizeof(uint8_t); // uw3_num_nonsparse stoage
	shared_mem_amount += 3 * X.size(1) * X.size(1) * sizeof(uint8_t); // uw3_nonsparse_indices storage

	sparse_symmetric_contraction_kernel<<<block_dim, grid, shared_mem_amount>>>(
		UW3_nonsparse_indices.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
		UW3_num_nonsparse.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
		UW3.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		UW1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

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
	int nthreadZ = 1)
{

	auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    dim3 block_dim(X.size(0));

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t shared_mem_amount =  nthreadX * X.size(1) * sizeof(float); // X storage
	shared_mem_amount +=  nthreadX * X.size(1) * sizeof(float); // grad stoage
	shared_mem_amount +=  nthreadX * sizeof(float); // output stoage
	shared_mem_amount += X.size(1) * X.size(1) * sizeof(uint8_t); // uw3_num_nonsparse stoage
	shared_mem_amount += 3 * X.size(1) * X.size(1) * sizeof(uint8_t); // uw3_nonsparse_indices storage

	sparse_symmetric_contraction_derivative_kernel<<<block_dim, grid, shared_mem_amount>>>(
		UW3_nonsparse_indices.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
		UW3_num_nonsparse.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
		UW3.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW3_deriv_factors.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		UW1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
		grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}



__global__ void sparse_full_symmetric_contraction_kernel(
	const torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> U3_nonsparse_indices,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U3_num_nonsparse,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U3_nonsparse_elements,
	const torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> U2_nonsparse_indices,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U2_num_nonsparse,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U2_nonsparse_elements,
	const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> U1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> W3,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> W2,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> W1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out)
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

	//load Us into shared memory...

	//non_sparse_indices U3: 16x16x3 for kdx, 16x16x3 for ldx
	//num non_sparse U3: 16x16

	// non_sparse_indices U2: 16x16
	for (int i = threadIdx.y; i < nl; i+=blockDim.y) {
		for (int j = threadIdx.x; j< nl; j+=blockDim.x) {
			
			int num_nonsparse_u3 = U3_num_nonsparse[i][j];

			buffer_u3_nonzeros[i][j] =  num_nonsparse_u3;

			for (int k = 0; k < num_nonsparse_u3; k ++) { 
				
				buffer_u3_kdx_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices_kdx[i][j][0][k]; // could re-order this to make it more efficient (j index last index)
				buffer_u3_ldx_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices_ldx[i][j][1][k];  // U3_nonsparse_indices_ldx[i][1][k][j]; 

				buffer_u3_values[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_elements[i][j][k]; // U3_nonsparse_elements[i][k][j];
			}

			int u2_kdx = U2_nonsparse_indices[i][j];

			buffer_u2_kdx_indices[i * nl + j] = u2_kdx;
			buffer_u2_values[i * nl + j] = U2_nonsparse_elements[i][j];
		}

		buffer_u1_values[i] = U1[i];
	}

	__syncthreads();

	for (int atom_id = blockIdx.x; atom_id < natoms; atom_id +=gridDim.x) {

		int element = atom_types[atom_id];

		for (int channel_id = threadIdx.x; channel_id < X.size(2); channel_id += blockDim.x){
			
			//load W3, W2, W1 into shared memory

			for (int i = threadIdx.y; i <W3.size(1); i += blockDim.y) {
				buffer_W3[i * blockDim.x + threadIdx.x] = W3[element][i][channel_id]; 
			}

			for (int i = threadIdx.y; i <W2.size(1); i += blockDim.y) {
				buffer_W2[i * blockDim.x + threadIdx.x] = W2[element][i][channel_id]; 
			}

			for (int i = threadIdx.y; i <W1.size(1); i += blockDim.y) {
				buffer_W1[i * blockDim.x + threadIdx.x] = W1[element][i][channel_id]; 
			}

			__syncthreads();

			//load X into shared memory

			for (int i = threadIdx.y; i <nl; i += blockDim.y) {
				buffer_X[i * blockDim.x + threadIdx.x] =  X[atom_id][i][channel_id];
			}

			//zero out operating buffer

			buffer_out[threadIdx.x] = 0.0;
			
			__syncthreads();

			float output_1 = 0.0;

			for (int i = threadIdx.y; i < nl; i += blockDim.y) {
				
				float Xi = buffer_X[i * blockDim.x + threadIdx.x]; 
				float uw1_i =  buffer_U1[i];
				float w1_i = buffer_W1[threadIdx.x];

				float output_2 = 0.0;

				for (int j = 0; j < nl; j +=1) {

					float Xj = buffer_X[j * blockDim.x + threadIdx.x]; 
					float U_ij = buffer_U2[i][j];
					uint8_t u2_k = buffer_U2_index[i][j];
					float w2 = buffer_W2[u2_k * blockDim.x + threadIdx.x];

					uint8_t uw3_num_nonsparse = buffer_uw3_num_nonsparse[i * nl + j];

					float uw2_ij  = w2 * u2_k;

					float output_3 = 0.0;

					for (uint8_t k = 0; k < uw3_num_nonsparse; k ++) {

						uint8_t kdx = buffer_uw3_nonsparse_indices[i * (nl * 3) + (k * nl) + j];
						
						float Xk = buffer_X[kdx * blockDim.x + threadIdx.x];

						float w3 = buffer_W3[ldx * blockDim.x + threadIdx.x];

						float uw3_ijk = u3 * w3;

						output_3 += uw3_ijk * Xk;

					}

					output_2 += (output_3 + uw2_ij) * Xj;
				}

				output_1 += (output_2 + uw1_i) * Xi;

			}
			
			atomicAdd(&buffer_out[threadIdx.x], output_1); 

			__syncthreads();

			if (threadIdx.y == 0) {
				out[atom_id][channel_id] = buffer_out[threadIdx.x];
			}
		}
	}
}

void sparse_full_symmetric_contraction_gpu(
	torch::Tensor U3_nonsparse_indices,
	torch::Tensor U3_num_nonsparse,
	torch::Tensor U3_nonsparse_elements,
	torch::Tensor U2_nonsparse_indices,
	torch::Tensor U2_num_nonsparse,
	torch::Tensor U2_nonsparse_elements,
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
	int nthreadZ = 1) {

	dim3 block_dim(X.size(0));

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t shared_mem_amount =  nthreadX * X.size(1) * sizeof(float); // X storage
	shared_mem_amount +=  nthreadX * sizeof(float); // output stoage

	shared_mem_amount += X.size(1) * X.size(1) * sizeof(uint8_t); // U3_nonsparse_indices stoage
	shared_mem_amount += 3 * X.size(1) * X.size(1) * sizeof(uint8_t); // U3_num_nonsparse storage
	shared_mem_amount += 3 * X.size(1) * X.size(1) * sizeof(float); // U3_nonsparse_elements storage

	shared_mem_amount += X.size(1) * X.size(1) * sizeof(uint8_t); // U3_nonsparse_indices stoage
	shared_mem_amount += 3 * X.size(1) * X.size(1) * sizeof(uint8_t); // U3_num_nonsparse storage
	shared_mem_amount += 3 * X.size(1) * X.size(1) * sizeof(float); // U3_nonsparse_elements storage

	shared_mem_amount += W3.shape(1) * nthreadX * sizeof(float); // W3 storage
	shared_mem_amount += W2.shape(1) * nthreadX * sizeof(float); // W2 storage
	shared_mem_amount += W1.shape(1) * nthreadX * sizeof(float); // W1 storage

	sparse_full_symmetric_contraction_kernel<<<block_dim, grid, shared_mem_amount>>>(
		U3_nonsparse_indices.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
		U3_num_nonsparse.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
		U3_nonsparse_elements.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		U2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		U1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		W3.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		W2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		W1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());


	return out;
}


