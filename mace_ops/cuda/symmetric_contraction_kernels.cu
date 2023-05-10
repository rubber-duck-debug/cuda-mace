#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define FULL_MASK 0xffffffff

__global__ void sparse_full_symmetric_contraction_kernel(
	const torch::PackedTensorAccessor32<uint8_t, 4, torch::RestrictPtrTraits> U3_nonsparse_indices,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U3_num_nonsparse,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> U3_nonsparse_elements,

	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U2_nonsparse_indices,
	const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> U2_nonsparse_elements,

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
	const int u3_maxn_nonsparse = U3_nonsparse_indices.size(3);

	size_t offset = 0;

	float *buffer_X = reinterpret_cast<float *>(buffer + offset);
	offset += blockDim.x * nl * sizeof(float);
	float *buffer_out = reinterpret_cast<float *>(buffer + offset);
	offset += blockDim.x * sizeof(float);

	/** U3 storage buffers **/
	uint8_t *buffer_u3_kdx_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(uint8_t);
	uint8_t *buffer_u3_ldx_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(uint8_t);
	uint8_t *buffer_u3_nonzeros = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += nl * nl * sizeof(uint8_t);
	float *buffer_u3_values = reinterpret_cast<float *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(float);
	/** U2 storage buffers **/
	uint8_t *buffer_u2_kdx_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += nl * nl * sizeof(uint8_t);
	float *buffer_u2_values = reinterpret_cast<float *>(buffer + offset);
	offset += nl * nl * sizeof(float);
	/** U1 storage buffers **/
	float *buffer_u1_values = reinterpret_cast<float *>(buffer + offset);
	offset += nl * sizeof(float);
	/** weights storage buffers **/
	float *buffer_W3 = reinterpret_cast<float *>(buffer + offset);
	offset += W3.size(1) * blockDim.x * sizeof(float);
	float *buffer_W2 = reinterpret_cast<float *>(buffer + offset);
	offset += W2.size(1) * blockDim.x * sizeof(float);
	float *buffer_W1 = reinterpret_cast<float *>(buffer + offset);
	offset += W1.size(1) * blockDim.x * sizeof(float);

	// load Us into shared memory...

	// non_sparse_indices U3: 16x16x3 for kdx, 16x16x3 for ldx
	// num non_sparse U3: 16x16

	// non_sparse_indices U2: 16x16
	for (int i = threadIdx.y; i < nl; i += blockDim.y)
	{
		for (int j = threadIdx.x; j < nl; j += blockDim.x)
		{

			int num_nonsparse_u3 = U3_num_nonsparse[i][j];

			buffer_u3_nonzeros[i * nl + j] = num_nonsparse_u3;

			for (int k = 0; k < num_nonsparse_u3; k++)
			{

				buffer_u3_kdx_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices[i][j][0][k]; // could re-order this to make it more efficient (j index last index)
				buffer_u3_ldx_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices[i][j][1][k]; // U3_nonsparse_indices_ldx[i][1][k][j];

				buffer_u3_values[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_elements[i][j][k]; // U3_nonsparse_elements[i][k][j];
			}

			int u2_kdx = U2_nonsparse_indices[i][j];

			buffer_u2_kdx_indices[i * nl + j] = u2_kdx;
			buffer_u2_values[i * nl + j] = U2_nonsparse_elements[i][j];
		}

		buffer_u1_values[i] = U1[i];
	}

	__syncthreads();

	for (int atom_id = blockIdx.x; atom_id < natoms; atom_id += gridDim.x)
	{

		int element = atom_types[atom_id];

		for (int channel_id = threadIdx.x; channel_id < X.size(2); channel_id += blockDim.x)
		{

			/** load W3, W2, W1 into shared memory **/
			for (int i = threadIdx.y; i < W3.size(1); i += blockDim.y)
			{
				buffer_W3[i * blockDim.x + threadIdx.x] = W3[element][i][channel_id];
			}

			for (int i = threadIdx.y; i < W2.size(1); i += blockDim.y)
			{
				buffer_W2[i * blockDim.x + threadIdx.x] = W2[element][i][channel_id];
			}

			for (int i = threadIdx.y; i < W1.size(1); i += blockDim.y)
			{
				buffer_W1[i * blockDim.x + threadIdx.x] = W1[element][i][channel_id];
			}

			__syncthreads();

			/** load X into shared memory **/
			for (int i = threadIdx.y; i < nl; i += blockDim.y)
			{
				buffer_X[i * blockDim.x + threadIdx.x] = X[atom_id][i][channel_id];
			}

			// zero out operating buffer
			buffer_out[threadIdx.x] = 0.0;

			__syncthreads();

			float output_1 = 0.0;

			for (int i = threadIdx.y; i < nl; i += blockDim.y)
			{

				float Xi = buffer_X[i * blockDim.x + threadIdx.x];

				float u1_i = buffer_u1_values[i];
				float w1_i = buffer_W1[threadIdx.x];

				float uw1_i = u1_i * w1_i;

				float output_2 = 0.0;

				for (int j = 0; j < nl; j++)
				{

					float Xj = buffer_X[j * blockDim.x + threadIdx.x];

					float u2_ij = buffer_u2_values[i * nl + j];
					uint8_t u2_kdx = buffer_u2_kdx_indices[i * nl + j];
					float w2 = buffer_W2[u2_kdx * blockDim.x + threadIdx.x];

					float uw2_ij = w2 * u2_ij;

					uint8_t uw3_num_nonsparse = buffer_u3_nonzeros[i * nl + j];

					float output_3 = 0.0;

					for (uint8_t k = 0; k < uw3_num_nonsparse; k++)
					{

						uint8_t u3_kdx = buffer_u3_kdx_indices[i * (nl * 3) + (k * nl) + j];
						uint8_t u3_ldx = buffer_u3_ldx_indices[i * (nl * 3) + (k * nl) + j];

						float w3 = buffer_W3[u3_ldx * blockDim.x + threadIdx.x];
						float u3_ijkdx = buffer_u3_values[i * (nl * 3) + (k * nl) + j];

						float Xk = buffer_X[u3_kdx * blockDim.x + threadIdx.x];

						float uw3_ijk = u3_ijkdx * w3;

						output_3 += uw3_ijk * Xk;
					}

					output_2 += (output_3 + uw2_ij) * Xj;
				}

				output_1 += (output_2 + uw1_i) * Xi;
			}

			atomicAdd(&buffer_out[threadIdx.x], output_1);

			__syncthreads();

			if (threadIdx.y == 0)
			{
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
	torch::Tensor U2_nonsparse_elements,
	torch::Tensor U1,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor out,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1)
{

	const int natoms = X.size(0);
	const int nl = X.size(1);
	const int nchannels = X.size(2);
	const int u3_n_nonsparse = U3_nonsparse_indices.size(3);

	dim3 block_dim(natoms);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t shared_mem_amount = nthreadX * nl * sizeof(float); // X storage
	shared_mem_amount += nthreadX * sizeof(float);			  // output stoage

	shared_mem_amount += 2 * nl * nl * u3_n_nonsparse * sizeof(uint8_t); // U3_nonsparse_indices stoage for kdx and ldx
	shared_mem_amount += nl * nl * sizeof(uint8_t);						 // U3_num_nonsparse storage
	shared_mem_amount += u3_n_nonsparse * nl * nl * sizeof(float);		 // U3_nonsparse_elements storage

	shared_mem_amount += nl * nl * sizeof(uint8_t); // U3_nonsparse_indices stoage
	shared_mem_amount += nl * nl * sizeof(float);	// U3_nonsparse_elements storage

	shared_mem_amount += W3.size(1) * nthreadX * sizeof(float); // W3 storage
	shared_mem_amount += W2.size(1) * nthreadX * sizeof(float); // W2 storage
	shared_mem_amount += W1.size(1) * nthreadX * sizeof(float); // W1 storage

	// printf("shared mem request: %d\n", shared_mem_amount);

	sparse_full_symmetric_contraction_kernel<<<block_dim, grid, shared_mem_amount>>>(
		U3_nonsparse_indices.packed_accessor32<uint8_t, 4, torch::RestrictPtrTraits>(),
		U3_num_nonsparse.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
		U3_nonsparse_elements.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		U2_nonsparse_indices.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
		U2_nonsparse_elements.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
		U1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
		W3.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		W2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		W1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

template <typename scalar_t>
__global__ void sparse_full_symmetric_contraction_derivative_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,

	const torch::PackedTensorAccessor32<uint8_t, 4, torch::RestrictPtrTraits> U3_nonsparse_indices,
	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U3_num_nonsparse,
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U3_nonsparse_elements,

	const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> U2_nonsparse_indices,
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> U2_nonsparse_elements,

	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> U1,

	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> W3,
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> W2,
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> W1,

	torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_out,
	const bool requires_grad)
{

	extern __shared__ char buffer[];

	const int natoms = X.size(0);
	const int nl = 16;
	const int nchannels = X.size(2);
	const int u3_maxn_nonsparse = U3_nonsparse_indices.size(3);

	size_t offset = 0;

	scalar_t *buffer_X = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += blockDim.x * nl * sizeof(scalar_t);
	scalar_t *buffer_out = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += blockDim.x * sizeof(scalar_t);

	scalar_t *buffer_u3_values = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(scalar_t);
	scalar_t *buffer_u2_values = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += nl * nl * sizeof(scalar_t);
	scalar_t *buffer_u1_values = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += nl * sizeof(scalar_t);

	scalar_t *buffer_W3 = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += W3.size(1) * blockDim.x * sizeof(scalar_t);
	scalar_t *buffer_W2 = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += W2.size(1) * blockDim.x * sizeof(scalar_t);
	scalar_t *buffer_W1 = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += W1.size(1) * blockDim.x * sizeof(scalar_t);

	/** U3 storage buffers **/
	uint8_t *buffer_u3_kdx_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(uint8_t);
	uint8_t *buffer_u3_ldx1_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(uint8_t);
	uint8_t *buffer_u3_ldx2_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(uint8_t);
	uint8_t *buffer_u3_ldx3_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += u3_maxn_nonsparse * nl * nl * sizeof(uint8_t);

	/** U2 storage buffers **/
	uint8_t *buffer_u3_nonzeros = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += nl * nl * sizeof(uint8_t);
	uint8_t *buffer_u2_kdx_indices = reinterpret_cast<uint8_t *>(buffer + offset);
	offset += nl * nl * sizeof(uint8_t);

	// load Us into shared memory...

	// non_sparse_indices U3: 16x16x3 for kdx, 16x16x3 for ldx
	// num non_sparse U3: 16x16

	// non_sparse_indices U2: 16x16

	for (int atom_id = blockIdx.x; atom_id < natoms; atom_id += gridDim.x)
	{

		int element = atom_types[atom_id];


		for (int i = threadIdx.y; i < nl; i += blockDim.y)
		{
			for (int j = threadIdx.x; j < nl; j += blockDim.x)
			{

			int num_nonsparse_u3 = U3_num_nonsparse[i][j];

			buffer_u3_nonzeros[i * nl + j] = num_nonsparse_u3;

			for (int k = 0; k < num_nonsparse_u3; k++)
			{

				buffer_u3_kdx_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices[i][j][0][k];
				buffer_u3_ldx1_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices[i][j][1][k]; // U3_nonsparse_indices_ldx[i][1][k][j];

				/* derivative indices */
				buffer_u3_ldx2_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices[i][j][2][k];
				buffer_u3_ldx3_indices[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_indices[i][j][3][k];

				buffer_u3_values[i * (nl * 3) + (k * nl) + j] = U3_nonsparse_elements[i][j][k]; // U3_nonsparse_elements[i][k][j];
			}

			int u2_kdx = U2_nonsparse_indices[i][j];

			buffer_u2_kdx_indices[i * nl + j] = u2_kdx;
			buffer_u2_values[i * nl + j] = U2_nonsparse_elements[i][j];
		}

		buffer_u1_values[i] = U1[i][0];
		}

		__syncthreads();

		for (int channel_id = threadIdx.x; channel_id < X.size(2); channel_id += blockDim.x)
		{

			/** load X into shared memory **/
			for (int i = threadIdx.y; i < nl; i += blockDim.y)
			{
				buffer_X[i * blockDim.x + threadIdx.x] = X[atom_id][i][channel_id];
			}

			/** load W3, W2, W1 into shared memory **/
			for (int i = threadIdx.y; i < W3.size(1); i += blockDim.y)
			{
				buffer_W3[i * blockDim.x + threadIdx.x] = W3[element][i][channel_id];
			}

			for (int i = threadIdx.y; i < W2.size(1); i += blockDim.y)
			{
				buffer_W2[i * blockDim.x + threadIdx.x] = W2[element][i][channel_id];
			}

			for (int i = threadIdx.y; i < W1.size(1); i += blockDim.y)
			{
				buffer_W1[i * blockDim.x + threadIdx.x] = W1[element][i][channel_id];
			}

			buffer_out[threadIdx.x] = 0.0;

			__syncthreads();

			float output_1 = 0.0;


			for (int i = threadIdx.y; i < nl; i += blockDim.y)
			{

				float Xi = buffer_X[i * blockDim.x + threadIdx.x];

				float u1_i = buffer_u1_values[i];
				float w1_i = buffer_W1[threadIdx.x];

				float uw1_i = u1_i * w1_i;

				float deriv1_tmp = uw1_i;

				float output_2 = 0.0;

				#pragma unroll
				for (int j = 0; j < nl; j++)
				{

					float Xj = buffer_X[j * blockDim.x + threadIdx.x];

					float u2_ij = buffer_u2_values[i * nl + j];
					uint8_t u2_kdx = buffer_u2_kdx_indices[i * nl + j];
					float w2 = buffer_W2[u2_kdx * blockDim.x + threadIdx.x];

					float uw2_ij = w2 * u2_ij;

					float deriv_1_j_tmp = uw2_ij;

					uint8_t uw3_num_nonsparse = buffer_u3_nonzeros[i * nl + j];

					float output_3 = 0.0;

					for (uint8_t k = 0; k < uw3_num_nonsparse; k++)
					{

						uint8_t u3_kdx = buffer_u3_kdx_indices[i * (nl * 3) + (k * nl) + j];

						uint8_t u3_ldx1 = buffer_u3_ldx1_indices[i * (nl * 3) + (k * nl) + j];
						uint8_t u3_ldx2 = buffer_u3_ldx2_indices[i * (nl * 3) + (k * nl) + j];
						uint8_t u3_ldx3 = buffer_u3_ldx3_indices[i * (nl * 3) + (k * nl) + j];

						float w3_1 = buffer_W3[u3_ldx1 * blockDim.x + threadIdx.x];
						float w3_2 = buffer_W3[u3_ldx2 * blockDim.x + threadIdx.x];
						float w3_3 = buffer_W3[u3_ldx3 * blockDim.x + threadIdx.x];

						float u3_ijkdx = buffer_u3_values[i * (nl * 3) + (k * nl) + j];

						float Xk = buffer_X[u3_kdx * blockDim.x + threadIdx.x];

						float uw3_ijk = u3_ijkdx * w3_1;

						output_3 += uw3_ijk * Xk;

						deriv_1_j_tmp += u3_ijkdx * (w3_1 + w3_2 + w3_3) * Xk;
					}

					output_2 += (output_3 + uw2_ij) * Xj;

					deriv1_tmp += (uw2_ij + deriv_1_j_tmp) * Xj;
				}

				output_1 += (output_2 + uw1_i) * Xi;

				grad_out[atom_id][i][channel_id] = deriv1_tmp;
			}

			atomicAdd(&buffer_out[threadIdx.x], output_1);

			__syncthreads();

			if (threadIdx.y == 0)
			{
				out[atom_id][channel_id] = buffer_out[threadIdx.x];
			}
		}
	}
}

std::vector<torch::Tensor> sparse_full_symmetric_contraction_derivative_gpu(
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor U3_nonsparse_indices,
	torch::Tensor U3_num_nonsparse,
	torch::Tensor U3_nonsparse_elements,
	torch::Tensor U2_nonsparse_indices,
	torch::Tensor U2_nonsparse_elements,
	torch::Tensor U1,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1)
{

	torch::Tensor output = torch::empty({X.size(0), X.size(2)},
										torch::TensorOptions()
											.dtype(X.dtype())
											.device(X.device()));

	torch::Tensor grad;

	if (X.requires_grad())
	{
		grad = torch::empty_like(X);
	}
	else
	{
		grad = torch::empty({1, 1, 1},
							torch::TensorOptions()
								.dtype(X.dtype())
								.device(X.device()));
	}

	const int natoms = X.size(0);
	const int nl = X.size(1);
	const int nchannels = X.size(2);
	const int u3_n_nonsparse = U3_nonsparse_indices.size(3);

	dim3 block_dim(natoms);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	AT_DISPATCH_FLOATING_TYPES(
		X.type(), "sparse_full_symmetric_contraction_derivative_gpu", ([&]
																	   {
            size_t total_buff_size = 0;

            size_t shared_mem_amount =  nthreadX * nl * sizeof(scalar_t); // X storage
			shared_mem_amount +=  nthreadX * sizeof(scalar_t); // output stoage

			shared_mem_amount += 4 * nl * nl * u3_n_nonsparse * sizeof(uint8_t); // U3_nonsparse_indices stoage for kdx and ldx
			shared_mem_amount += nl * nl * sizeof(uint8_t); // U3_num_nonsparse storage
			shared_mem_amount += u3_n_nonsparse * nl * nl * sizeof(scalar_t); // U3_nonsparse_elements storage

			shared_mem_amount += nl * nl * sizeof(uint8_t); // U3_nonsparse_indices stoage
			shared_mem_amount += nl * nl * sizeof(scalar_t); // U3_nonsparse_elements storage

			shared_mem_amount += W3.size(1) * nthreadX * sizeof(scalar_t); // W3 storage
			shared_mem_amount += W2.size(1) * nthreadX * sizeof(scalar_t); // W2 storage
			shared_mem_amount += W1.size(1) * nthreadX * sizeof(scalar_t); // W1 storage

            sparse_full_symmetric_contraction_derivative_kernel<<<block_dim, grid, shared_mem_amount>>>(
				X.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
				U3_nonsparse_indices.packed_accessor32<uint8_t, 4, torch::RestrictPtrTraits>(),
				U3_num_nonsparse.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
				U3_nonsparse_elements.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				U2_nonsparse_indices.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
				U2_nonsparse_elements.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				U1.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				W3.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				W2.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				W1.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				X.requires_grad()); }));

	cudaDeviceSynchronize();

	return {output, grad};
}

template <typename scalar_t>
__global__ void symm_contraction_backward_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_X,
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
	torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output)
{

	extern __shared__ char buffer[];

	const int natoms = grad_X.size(0);
	const int nl = grad_X.size(1);
	const int nchannels = grad_X.size(2);

	int atom_idx = blockIdx.x;

	for (int channel_id = threadIdx.x; channel_id < nchannels; channel_id += blockDim.x)
	{

		scalar_t grad = grad_input[atom_idx][channel_id];

		for (int l_id = threadIdx.y; l_id < nl; l_id += blockDim.y)
		{
			grad_output[atom_idx][l_id][channel_id] = grad * grad_X[atom_idx][l_id][channel_id];
		}
	}
}

torch::Tensor symm_contraction_backward(
	torch::Tensor gradX,
	torch::Tensor grad_input,
	int nthreadX = 64,
	int nthreadY = 1,
	int nthreadZ = 1)
{

	torch::Tensor output = torch::empty_like(gradX);

	const int natoms = output.size(0);

	dim3 block_dim(natoms);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	AT_DISPATCH_FLOATING_TYPES(
		gradX.type(), "symm_contraction_backward", ([&]
													{ symm_contraction_backward_kernel<<<block_dim, grid>>>(gradX.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
																											grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
																											output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

	cudaDeviceSynchronize();

	return output;
}

class SymmetricContractionAutograd : public Function<SymmetricContractionAutograd>
{
public:
	static torch::Tensor forward(
		AutogradContext *ctx,
		torch::Tensor X,
		torch::Tensor atom_types,
		torch::Tensor U3_nonsparse_indices,
		torch::Tensor U3_num_nonsparse,
		torch::Tensor U3_nonsparse_elements,
		torch::Tensor U2_nonsparse_indices,
		torch::Tensor U2_nonsparse_elements,
		torch::Tensor U1,
		torch::Tensor W3,
		torch::Tensor W2,
		torch::Tensor W1,
		const int64_t nthreadx,
		const int64_t nthready,
		const int64_t nthreadz)
	{

		auto result = sparse_full_symmetric_contraction_derivative_gpu(
			X,
			atom_types,
			U3_nonsparse_indices,
			U3_num_nonsparse,
			U3_nonsparse_elements,
			U2_nonsparse_indices,
			U2_nonsparse_elements,
			U1,
			W3,
			W2,
			W1,
			nthreadx,
			nthready,
			nthreadz);

		if (X.requires_grad())
		{
			ctx->saved_data["nthreadx"] = nthreadx;
			ctx->saved_data["nthready"] = nthready;
			ctx->saved_data["nthreadz"] = nthreadz;

			ctx->save_for_backward({result[1]});
		}

		return result[0];
	}

	static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
	{
		auto saved_variables = ctx->get_saved_variables();

		auto gradX = saved_variables[0];

		int nthreadx = ctx->saved_data["nthreadx"].toInt();
		int nthready = ctx->saved_data["nthready"].toInt();
		int nthreadz = ctx->saved_data["nthreadz"].toInt();

		torch::Tensor result = symm_contraction_backward(gradX, grad_outputs[0], nthreadx, nthready, nthreadz);

		torch::Tensor undef;

		return {result, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef};
	}
};

torch::Tensor symmetric_contraction(
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor U3_nonsparse_indices,
	torch::Tensor U3_num_nonsparse,
	torch::Tensor U3_nonsparse_elements,
	torch::Tensor U2_nonsparse_indices,
	torch::Tensor U2_nonsparse_elements,
	torch::Tensor U1,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	const int64_t nthreadx,
	const int64_t nthready,
	const int64_t nthreadz)
{

	return SymmetricContractionAutograd::apply(
		X,
		atom_types,
		U3_nonsparse_indices,
		U3_num_nonsparse,
		U3_nonsparse_elements,
		U2_nonsparse_indices,
		U2_nonsparse_elements,
		U1,
		W3,
		W2,
		W1,
		nthreadx,
		nthready,
		nthreadz);
}

TORCH_LIBRARY(mace_cuda_symm_contraction, m)
{
	m.def("symmetric_contraction", &symmetric_contraction);
}
