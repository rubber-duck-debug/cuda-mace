#include <math.h>
#include <torch/torch.h>
#include <iostream>

using namespace std;

#define FULL_MASK 0xffffffff

__global__ void correlation_3_main_kernel(
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW,
	const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> UW_nonsparse_indices,
	const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> UW_num_nonzeros,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad,
	bool requires_grad)
{

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_out = reinterpret_cast<float *>(buffer + offset);

	offset += blockDim.x * blockDim.y * sizeof(float);

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

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx += n_blocks_x)
	{

		int element = atom_types[atom_idx];

		for (int i = 0; i < n_iter_x; i++)
		{

			int idx_i = i * n_threads_x + tx;

			for (int ix = by; ix < UW.size(0); ix += n_blocks_y)
			{

				for (int iy = ty; iy < UW.size(1); iy += n_threads_y)
				{

					int num_nonsparse = UW_num_nonzeros[ix][iy];
					// int num_nonsparse = 3;

					for (int l = 0; l < num_nonsparse; l++)
					{

						int ldx = UW_nonsparse_indices[ix][iy][l]; // these cost a lot!
						// int ldx = 0;
						float uw = UW[ix][iy][ldx][element][idx_i]; // these cost a lot!
						// float uw = 0.0;

						// float x = buffer_x[ldx * blockDim.x + threadIdx.x];
						float x = X[atom_idx][ldx][idx_i];

						atomicAdd(&out[atom_idx][ix][iy][idx_i], uw * x);

						if (requires_grad)
						{
							atomicAdd(&grad[atom_idx][ldx][idx_i], uw);
						}
					}
					// out[atom_idx][ix][iy][idx_i] = buffer_out[ty * blockDim.x + threadIdx.x];
				}

				// nthreads_x = 32, nthreads_y = 4

				//  0  1  2  3  4  5  6  7 ... 31,  0
				// 32 33 34 35 36 37 38 39 ... 63,  1
				// 64 65 66 67 68 69 70 71 ... 95,  2
				// 96 ...                  ... 127, 3
			}
		}
	}
}

__global__ void correlation_3_main_and_grad_kernel(
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW,
	const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> UW_nonsparse_indices,
	const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> UW_num_nonzeros,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad,
	bool requires_grad)
{

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_out = reinterpret_cast<float *>(buffer + offset);

	offset += blockDim.x * blockDim.y * blockDim.z * sizeof(float);

	float *grad_out;

	if (requires_grad)
	{
		grad_out = reinterpret_cast<float *>(buffer + offset);
		offset += blockDim.x * blockDim.y * blockDim.z * sizeof(float);
	}

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

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx += n_blocks_x)
	{

		int element = atom_types[atom_idx];

		for (int i = 0; i < n_iter_x; i++)
		{

			int idx_i = i * n_threads_x + tx;

			for (int ix = tz; ix < UW.size(0); ix += n_threads_z)
			{

				for (int iy = ty; iy < UW.size(1); iy += n_threads_y)
				{

					int num_nonsparse = UW_num_nonzeros[ix][iy];

					for (int l = 0; l < num_nonsparse; l++)
					{

						int ldx = UW_nonsparse_indices[ix][iy][l];
						float uw = UW[ix][iy][ldx][element][idx_i];
						float x = X[atom_idx][ldx][idx_i];

						atomicAdd(&out[atom_idx][ix][iy][idx_i], uw * x);

						if (requires_grad)
						{
							atomicAdd(&grad[atom_idx][ldx][idx_i], uw);
						}
					}
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
	torch::Tensor grad,
	bool requires_grad,
	int nblockX = 1,
	int nblockY = 1,
	int nblockZ = 1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1)
{

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t buffer_size = 0;

	buffer_size += grid.x * grid.y * sizeof(float);

	correlation_3_main_kernel<<<blocks, grid, buffer_size>>>(
		UW.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW_nonsparse_indices.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
		UW_nonsparse_num_nonzeros.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		requires_grad);

	cudaDeviceSynchronize();
}

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
	int nthreadZ = 1)
{

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t buffer_size = 0;

	// buffer_size += grid.x * 16  * sizeof(float);

	correlation_3_main_and_grad_kernel<<<blocks, grid, buffer_size>>>(
		UW.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW_nonsparse_indices.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
		UW_nonsparse_num_nonzeros.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		requires_grad);

	cudaDeviceSynchronize();
}

__global__ void correlation_2_contraction_kernel(
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> UW2,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> prev_layer,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_in,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_out,
	bool requires_grad)
{

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_c = reinterpret_cast<float *>(buffer + offset);

	offset += blockDim.x * UW2.size(0) * sizeof(float);

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

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx += n_blocks_x)
	{

		int element = atom_types[atom_idx];

		for (int i = 0; i < n_iter_x; i++)
		{

			int idx_i = i * n_threads_x + tx;

			for (int ix = ty; ix < UW2.size(0); ix += n_threads_y)
			{
				buffer_c[ix * blockDim.x + threadIdx.x] = 0.0;
			}

			__syncthreads();

			for (int iy = tz; iy < UW2.size(1); iy += n_threads_z)
			{

				float x = X[atom_idx][iy][idx_i];
				
				float gin = 0.0;

				if (requires_grad){
					gin = grad_in[atom_idx][iy][idx_i];
				}

				for (int ix = ty; ix < UW2.size(0); ix += n_threads_y)
				{

					float uw = UW2[ix][iy][element][idx_i];

					float sum_uw = uw +  prev_layer[atom_idx][ix][iy][idx_i];

					// atomicAdd(&out[atom_idx][ix][idx_i], uw * x);

					buffer_c[ix * blockDim.x + threadIdx.x] += sum_uw * x;

					if (requires_grad) {
						atomicAdd(&grad_out[atom_idx][iy][idx_i], sum_uw);
					}
				}

			}

			for (int ix = ty; ix < UW2.size(0); ix += n_threads_y)
			{
				out[atom_idx][ix][idx_i] = buffer_c[ix * blockDim.x + threadIdx.x];
			}

			
		}
	}
}

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
	int nthreadZ = 1)
{

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	correlation_2_contraction_kernel<<<blocks, grid, grid.x * UW2.size(0) * sizeof(float)>>>(
		UW2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		prev_layer.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		grad_in.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		requires_grad);

	cudaDeviceSynchronize();
}

__global__ void correlation_1_contraction_kernel(
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> UW1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> prev_layer,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_in,
	torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_out,
	bool requires_grad)
{

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_c = reinterpret_cast<float *>(buffer + offset);

	offset += blockDim.x * sizeof(float);

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

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx += n_blocks_x)
	{

		int element = atom_types[atom_idx];

		for (int i = 0; i < n_iter_x; i++)
		{

			int idx_i = i * n_threads_x + tx;

			buffer_c[threadIdx.x] = 0.0;

			for (int gy = ty; gy < UW1.size(0); gy += n_threads_y)
			{

				float uw = UW1[gy][element][idx_i] + prev_layer[atom_idx][gy][idx_i];

				float x = X[atom_idx][gy][idx_i];

				atomicAdd(&buffer_c[threadIdx.x], uw * x);

				// shared_memory here?
				// atomicAdd(&out[atom_idx][idx_i], uw * x);
			}

			out[atom_idx][idx_i] = buffer_c[threadIdx.x];
		}
	}
}

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
	int nthreadZ = 1)
{

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	correlation_1_contraction_kernel<<<blocks, grid, grid.x * sizeof(float)>>>(
		UW1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		prev_layer.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		grad_in.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
		grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		requires_grad);

	cudaDeviceSynchronize();
}


__global__ void symmetric_contraction_derivative_kernel(
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW3,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> UW2,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> UW1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_out)
{

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_X = reinterpret_cast<float *>(buffer + offset);

	offset += blockDim.x * X.size(1) * sizeof(float);

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

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx += n_blocks_x)
	{

		int element = atom_types[atom_idx];

		for (int r = threadIdx.x; r < X.size(2); r+= blockDim.x){
			
			for (int i = threadIdx.y; i < X.size(1); i += blockDim.y) {
				buffer_X[i * blockDim.x + threadIdx.x] =  X[atom_idx][i][r];
			}
			__syncthreads();

			for (int i = threadIdx.y; i < X.size(1); i +=blockDim.y) {

				float deriv1_tmp = UW1[i][element][r];

				float uw2_ij  = 0.0;
				
				for (int j = 0; j < X.size(1); j ++) {
					

					float uw2_ij = UW2[i][j][element][r];
					float uw2_ji =UW2[j][i][element][r];

					float  deriv_1_j_tmp = uw2_ij; // uw2_ij;
					float deriv_1_j_tmp2 = uw2_ij; // uw2_ji;

					float Xj = buffer_X[j * blockDim.x + threadIdx.x]; //X[atom_idx][j][r];

					for (int k = 0; k < X.size(1); k ++) {

						float Xk = buffer_X[k * blockDim.x + threadIdx.x];  // X[atom_idx][k][r];

						float uw3_ijk = UW3[i][j][k][element][r]; 
						float uw3_jki = UW3[j][k][i][element][r];
						float uw3_jik =UW3[j][i][k][element][r];


						deriv_1_j_tmp +=uw3_ijk * Xk;

						deriv_1_j_tmp2 +=uw3_jki* Xk +uw3_jik * Xk;

						//deriv_1_j_tmp2 += UW3[j][k][i][element][r] * Xk + UW3[i][j][k][element][r] * Xk;

					}

					deriv_1_j_tmp *= Xj;
					deriv_1_j_tmp2 *= Xj;

					deriv1_tmp +=  (deriv_1_j_tmp + deriv_1_j_tmp2);

				}

				grad_out[atom_idx][i][r] = deriv1_tmp;

			}
	}
	}
}


__global__ void sparse_symmetric_contraction_derivative_kernel(
	const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> UW3_nonsparse_indices,
	const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> UW2_num_nonsparse,
	const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW3,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> UW2,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> UW1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_out)
{

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_X = reinterpret_cast<float *>(buffer + offset);
	offset += blockDim.x * X.size(1) * sizeof(float);
	float *buffer_grad = reinterpret_cast<float *>(buffer + offset);

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

	for (int atom_idx = bx; atom_idx < X.size(0); atom_idx += n_blocks_x)
	{

		int element = atom_types[atom_idx];

		for (int r = threadIdx.x; r < X.size(2); r+= blockDim.x){
			
			for (int i = threadIdx.y; i < X.size(1); i += blockDim.y) {
				buffer_X[i * blockDim.x + threadIdx.x] =  X[atom_idx][i][r];
			}
			__syncthreads();

			for (int i = threadIdx.y; i < X.size(1); i +=blockDim.y) {

				float deriv1_tmp = UW1[i][element][r];


				for (int j = 0; j < X.size(1); j ++) {
			
					float uw2_ij  = UW2[i][j][element][r];
				
					float uw2_ji =UW2[j][i][element][r];

					int uw3_num_nonsparse = UW2_num_nonsparse[i][j];

					float  deriv_1_j_tmp = uw2_ij;
					float deriv_1_j_tmp2 = uw2_ij;

					float Xj = buffer_X[j * blockDim.x + threadIdx.x]; //X[atom_idx][j][r];

					for (int k = 0; k < uw3_num_nonsparse; k ++) {
						int kdx1 = UW3_nonsparse_indices[i][j][k];
						int kdx2 = UW3_nonsparse_indices[j][i][k];

						float Xk1 = buffer_X[kdx1 * blockDim.x + threadIdx.x];
						float Xk2 = buffer_X[kdx2 * blockDim.x + threadIdx.x];

						float uw3_ijk = UW3[i][j][kdx1][element][r]; 

						float uw3_jik =UW3[j][i][kdx2][element][r];

						deriv_1_j_tmp +=uw3_ijk * Xk1;
						deriv_1_j_tmp2 += uw3_jik * Xk2;

					}

					for (int k = 0; k < X.size(1); k ++) {

						float Xk = buffer_X[k * blockDim.x + threadIdx.x];  // X[atom_idx][k][r];

						float uw3_jki = UW3[j][k][i][element][r];
						

						deriv_1_j_tmp2 +=uw3_jki* Xk;

						//deriv_1_j_tmp2 += UW3[j][k][i][element][r] * Xk + UW3[i][j][k][element][r] * Xk;

					}

					deriv_1_j_tmp *= Xj;
					deriv_1_j_tmp2 *= Xj;

					deriv1_tmp +=  (deriv_1_j_tmp + deriv_1_j_tmp2);

				}

				buffer_grad[i * blockDim.x + threadIdx.x] = deriv1_tmp;

			}
	}
	}
}


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
	int nthreadZ = 1)
{

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t shared_mem_amount = nthreadX * X.size(1) * sizeof(float); // X storage

	symmetric_contraction_derivative_kernel<<<blocks, grid, shared_mem_amount>>>(
		UW3.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		UW1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

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
	int nthreadZ = 1)
{

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t shared_mem_amount = nthreadX * X.size(1) * sizeof(float); // X storage
	shared_mem_amount += nthreadX * X.size(1) * sizeof(float); // grad stoage

	sparse_symmetric_contraction_derivative_kernel<<<blocks, grid, shared_mem_amount>>>(
		UW3_nonsparse_indices.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
		UW3_num_nonsparse.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
		UW3.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		UW1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}