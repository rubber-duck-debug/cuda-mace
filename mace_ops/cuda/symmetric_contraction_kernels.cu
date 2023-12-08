#include <torch/script.h>
#include <cooperative_groups.h>
#include <iostream>
#include <cuda_fp16.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;
using namespace cooperative_groups;

#define FULL_MASK 0xffffffff

#define NATOMS_PER_BLOCK 4
#define WARP_SIZE 32

__device__ dim3 get_thread_indices(int nthreadsx, int nthreadsy, int nthreadsz)
{
	int thread_idx = threadIdx.x % nthreadsx;				// 0-16
	int thread_idy = threadIdx.x / (nthreadsx * nthreadsz); // 0-512 / 64 = 0-8
	int thread_idz = threadIdx.x / (nthreadsx * nthreadsy); // 0-512 / 128 = 0-4

	return dim3(thread_idx, thread_idy, thread_idz);
}

template <class T>
__host__ __device__ T *shared_array(std::size_t n_elements, void *&ptr,
									std::size_t *space = nullptr) noexcept
{
	const std::uintptr_t inptr = reinterpret_cast<uintptr_t>(ptr);
	const std::uintptr_t end = inptr + n_elements * sizeof(T);
	if (space)
		*space += static_cast<std::size_t>(end - inptr);
	ptr = reinterpret_cast<void *>(end);
	return reinterpret_cast<T *>(inptr);
}

template <typename scalar_t>
__global__ void symmetric_contraction_L0_forwards_new_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U3_num_nonzero,
	const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> U3_indices,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> U3_values,

	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_num_nonzero,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_indices,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> U2_values,

	const torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> U1_num_nonzero,
	const torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> U1_indices,

	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W3,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W2,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W1,
	const int w3_size,
	const int w2_size,
	const int w1_size,
	const int u3_maxn_nonsparse,
	torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
	const bool requires_grad)
{

	extern __shared__ char buffer[];
	void *sptr = buffer;

	const int natoms = X.size(0);
	const int nl = 16;
	const int nchannels = X.size(2);
	size_t space = 0;

	volatile scalar_t *buffer_X = shared_array<scalar_t>(WARP_SIZE * nl, sptr, &space);
	volatile scalar_t *buffer_out = shared_array<scalar_t>(WARP_SIZE * nl, sptr, &space);
	volatile scalar_t *buffer_grad = shared_array<scalar_t>(WARP_SIZE * nl, sptr, &space);

	volatile scalar_t *buffer_W3 = shared_array<scalar_t>(w3_size * WARP_SIZE, sptr, &space);
	volatile scalar_t *buffer_W2 = shared_array<scalar_t>(w2_size * WARP_SIZE, sptr, &space);
	volatile scalar_t *buffer_W1 = shared_array<scalar_t>(w1_size * WARP_SIZE, sptr, &space);

	volatile float *buffer_u3_values = shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr, &space);
	volatile float *buffer_u2_values = shared_array<float>(nl * nl, sptr, &space);

	volatile int *buffer_u3_indices = shared_array<int>(u3_maxn_nonsparse * nl * nl, sptr, &space);

	volatile short *buffer_u3_nonzeros = shared_array<short>(nl * nl, sptr, &space);
	volatile short *buffer_u2_nonzero = shared_array<short>(nl * nl, sptr, &space);
	volatile short *buffer_u2_indices = shared_array<short>(nl * nl, sptr, &space);

	int nthreadx = 16;
	int nthready = 8;
	int nthreadz = 1;

	dim3 thread_idx = get_thread_indices(nthreadx, nthready, nthreadz);

	for (int i = thread_idx.y; i < nl; i += nthready)
	{
		for (int j = thread_idx.x; j < nl; j += nthreadx)
		{
			int num_nonzero_u3 = U3_num_nonzero[0][i][j];

			buffer_u3_nonzeros[i * nl + j] = num_nonzero_u3;

			for (int k = 0; k < num_nonzero_u3; k++)
			{
				buffer_u3_indices[i * (nl * u3_maxn_nonsparse) + (k * nl) + j] = U3_indices[k][i][j]; // packed 32 bit integer containing 4 x uint8 indices
				buffer_u3_values[i * (nl * u3_maxn_nonsparse) + (k * nl) + j] = U3_values[k][i][j];
			}

			buffer_u2_nonzero[i * nl + j] = U2_num_nonzero[0][i][j];
			buffer_u2_indices[i * nl + j] = U2_indices[0][i][j];
			buffer_u2_values[i * nl + j] = U2_values[0][i][j];
		}
	}

	__syncthreads();

	int atom_id = blockIdx.x;

	int element = atom_types[atom_id];

	nthreadx = WARP_SIZE;
	nthready = 4;
	nthreadz = 1;

	thread_idx = get_thread_indices(nthreadx, nthready, 1);

	int channel_id = blockIdx.y * WARP_SIZE + thread_idx.x;

	__syncthreads();

	for (int i = thread_idx.y; i < w3_size; i += nthready)
	{

		buffer_W3[i * WARP_SIZE + thread_idx.x] = W3[0][element][i][channel_id];

		if (i < w2_size)
		{
			buffer_W2[i * WARP_SIZE + thread_idx.x] = W2[0][element][i][channel_id];
		}

		if (i < w1_size)
		{
			buffer_W1[i * WARP_SIZE + thread_idx.x] = W1[0][element][i][channel_id];
		}

		if (i < nl)
		{
			buffer_X[i * WARP_SIZE + thread_idx.x] = X[atom_id][i][channel_id];
			buffer_out[i * WARP_SIZE + thread_idx.x] = 0.0;
			buffer_grad[i * WARP_SIZE + thread_idx.x] = 0.0;
		}
	}

	__syncthreads();

	nthreadx = 8;
	nthready = 2;
	nthreadz = 8;

	thread_idx = get_thread_indices(nthreadx, nthready, nthreadz);

	for (int buffer_channel_id = thread_idx.z; buffer_channel_id < WARP_SIZE; buffer_channel_id += nthreadz)
	{
		scalar_t output_1 = 0.0;

		for (int i = thread_idx.y; i < nl; i += nthready)
		{
			scalar_t Xi = buffer_X[i * WARP_SIZE + buffer_channel_id];

			scalar_t uw1 = 0.0;

			if (i == 0)
			{
				uw1 = buffer_W1[buffer_channel_id];
			}

			scalar_t output_2 = 0.0;
			scalar_t deriv1_tmp = 0.0;

			if (requires_grad)
				deriv1_tmp = uw1;

			for (int j = thread_idx.x; j < nl; j += nthreadx)
			{
				scalar_t Xj = buffer_X[j * WARP_SIZE + buffer_channel_id]; // y dimension loads channels...

				scalar_t uw2 = 0.0;

				if (buffer_u2_nonzero[i * nl + j] > 0)
				{
					uw2 = buffer_u2_values[i * nl + j] * buffer_W2[buffer_u2_indices[i * nl + j] * WARP_SIZE + buffer_channel_id];
				}

				int uw3_num_nonzero = buffer_u3_nonzeros[i * nl + j];

				scalar_t output_3 = 0.0;
				scalar_t deriv_1_j_tmp = 0.0;

				if (requires_grad)
					deriv_1_j_tmp = uw2;

				for (int k = 0; k < uw3_num_nonzero; k++)
				{
					int u3_mem_idx = i * (nl * u3_maxn_nonsparse) + (k * nl) + j;

					int compressed_indices = buffer_u3_indices[u3_mem_idx];

					int u3_ldx1 = compressed_indices & 0xFF;
					int u3_kdx = (compressed_indices >> 8) & 0xFF;

					scalar_t w3_1 = buffer_W3[u3_ldx1 * WARP_SIZE + buffer_channel_id];

					scalar_t u3 = buffer_u3_values[u3_mem_idx];

					scalar_t Xk = buffer_X[u3_kdx * WARP_SIZE + buffer_channel_id];

					output_3 += u3 * w3_1 * Xk;

					if (requires_grad)
					{
						int u3_ldx3 = (compressed_indices >> 16) & 0xFF;
						int u3_ldx2 = (compressed_indices >> 24) & 0xFF;

						scalar_t w3_2 = buffer_W3[u3_ldx2 * WARP_SIZE + buffer_channel_id];
						scalar_t w3_3 = buffer_W3[u3_ldx3 * WARP_SIZE + buffer_channel_id];

						deriv_1_j_tmp += u3 * (w3_1 + w3_2 + w3_3) * Xk;
					}
				}
				output_2 += (output_3 + uw2) * Xj;
				deriv1_tmp += (uw2 + deriv_1_j_tmp) * Xj;
			}

			output_1 += (output_2 + uw1) * Xi;

			if (requires_grad)
			{
				for (int offset = nthreadx / 2; offset > 0; offset /= 2)
				{
					deriv1_tmp += __shfl_down_sync(FULL_MASK, deriv1_tmp, offset);
				}

				if (thread_idx.x == 0)
				{
					buffer_grad[i * WARP_SIZE + buffer_channel_id] = deriv1_tmp;
				}
			}
		}

		for (int offset = nthreadx / 2; offset > 0; offset /= 2)
		{
			output_1 += __shfl_down_sync(FULL_MASK, output_1, offset);
		}

		if (thread_idx.x == 0)
		{
			buffer_out[buffer_channel_id] = output_1;
		}
	}

	// reorganise threads so we can write contiguously...
	// 512 threads / 32 = 16 warps, so 16 x 32
	thread_idx = get_thread_indices(WARP_SIZE, 4, 1);

	// write out all gradients...
	if (requires_grad)
	{
		for (int i = thread_idx.y; i < nl; i += 4)
		{
			grad[atom_id][0][i][thread_idx.x] = buffer_grad[i * WARP_SIZE + thread_idx.x];
		}
	}
	// writeout output
	if (thread_idx.y == 0)
	{
		out[atom_id][0][channel_id] = buffer_out[thread_idx.x];
	}
}

std::vector<torch::Tensor> symmetric_contraction_L0_forwards_new_gpu(
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor U3_num_nonzero,
	torch::Tensor U3_indices,
	torch::Tensor U3_values,
	torch::Tensor U2_num_nonzero,
	torch::Tensor U2_indices,
	torch::Tensor U2_values,
	torch::Tensor U1_num_nonzero,
	torch::Tensor U1_indices,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	const int w3_size,
	const int w2_size,
	const int w1_size,
	torch::Tensor u3_n_nonsparse,
	const int64_t nthreadX = 32,
	const int64_t nthreadY = 4,
	const int64_t nthreadZ = 1)
{

	torch::Tensor output = torch::zeros({X.size(0), 1, X.size(2)},
										torch::TensorOptions()
											.dtype(X.dtype())
											.device(X.device()));
	torch::Tensor grad;

	if (X.requires_grad())
	{
		grad = torch::zeros({X.size(0), 1, X.size(1), X.size(2)},
							torch::TensorOptions()
								.dtype(X.dtype())
								.device(X.device()));
	}
	else
	{
		grad = torch::zeros({1, 1, 1, 1},
							torch::TensorOptions()
								.dtype(X.dtype())
								.device(X.device()));
	}

	const int natoms = X.size(0);
	const int nl = X.size(1);
	const int nchannels = X.size(2);

	auto find_num_blocks = [](int x, int bdim)
	{ return (x + bdim - 1) / bdim; };

	dim3 block_dim(natoms, nchannels / WARP_SIZE);
	// dim3 block_dim(natoms);

	dim3 grid(128, 1, 1);
	AT_DISPATCH_FLOATING_TYPES(
		X.type(), "symmetric_contraction_L0_forwards_new_gpu", ([&]
																{
            size_t shared_size = 0;

			void* sptr = nullptr;

			shared_array<scalar_t>(WARP_SIZE * nl, sptr, &shared_size);
			shared_array<scalar_t>(WARP_SIZE * nl, sptr, &shared_size);
			shared_array<scalar_t>(WARP_SIZE * nl, sptr, &shared_size);

			shared_array<scalar_t>(w3_size * WARP_SIZE, sptr, &shared_size);
			shared_array<scalar_t>(w2_size * WARP_SIZE, sptr, &shared_size);
			shared_array<scalar_t>(w1_size * WARP_SIZE, sptr, &shared_size);

			shared_array<float>(u3_n_nonsparse[0].item<int>() * nl * nl, sptr, &shared_size);
			shared_array<float>(nl * nl, sptr, &shared_size);

			shared_array<int>(u3_n_nonsparse[0].item<int>() * nl * nl, sptr, &shared_size);
			shared_array<short>(nl * nl, sptr, &shared_size);
			shared_array<short>(nl * nl, sptr, &shared_size);
			shared_array<short>(nl * nl, sptr, &shared_size);

			//printf("shared mem required L0: %d\n", shared_size);
            symmetric_contraction_L0_forwards_new_kernel<<<block_dim, grid, shared_size>>>(
				X.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
				U3_num_nonzero.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
				U3_indices.packed_accessor32<int, 3, torch::RestrictPtrTraits>(), // L=0 specific
				U3_values.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), // L=0 specific
				U2_num_nonzero.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
				U2_indices.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
				U2_values.packed_accessor32<float,3, torch::RestrictPtrTraits>(),
				U1_num_nonzero.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
				U1_indices.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
				W3.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				W2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				W1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				w3_size,
				w2_size,
				w1_size,
				u3_n_nonsparse[0].item<int>(),
				output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				X.requires_grad()); }));

	cudaDeviceSynchronize();

	return {output, grad};
}
template <typename scalar_t>
__global__ void
//__launch_bounds__(256, 4)
symmetric_contraction_L0_forwards_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U3_num_nonzero,
	const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> U3_indices,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> U3_values,

	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_num_nonzero,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_indices,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> U2_values,

	const torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> U1_num_nonzero,
	const torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> U1_indices,

	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W3,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W2,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W1,
	const int w3_size,
	const int w2_size,
	const int w1_size,
	const int u3_maxn_nonsparse,
	torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
	const bool requires_grad)
{

	extern __shared__ char buffer[];
	void *sptr = buffer;

	const int natoms = X.size(0);
	const int nl = 16;

	volatile scalar_t *buffer_X = shared_array<scalar_t>(blockDim.x * nl, sptr);
	volatile scalar_t *buffer_out = shared_array<scalar_t>(blockDim.y * blockDim.x, sptr);
	volatile scalar_t *buffer_W3 = shared_array<scalar_t>(w3_size * blockDim.x, sptr);
	volatile scalar_t *buffer_W2 = shared_array<scalar_t>(w2_size * blockDim.x, sptr);
	volatile scalar_t *buffer_W1 = shared_array<scalar_t>(w1_size * blockDim.x, sptr);

	volatile float *buffer_u3_values = shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr);
	volatile float *buffer_u2_values = shared_array<float>(nl * nl, sptr);

	volatile int *buffer_u3_indices = shared_array<int>(u3_maxn_nonsparse * nl * nl, sptr);

	volatile short *buffer_u3_nonzeros = shared_array<short>(nl * nl, sptr);
	volatile short *buffer_u2_nonzero = shared_array<short>(nl * nl, sptr);
	volatile short *buffer_u2_indices = shared_array<short>(nl * nl, sptr);

	__syncthreads();

	int channel_id = blockIdx.y * blockDim.x + threadIdx.x;

	int element = atom_types[blockIdx.x];

	for (int i = threadIdx.y; i < w3_size; i += blockDim.y)
	{
		buffer_W3[i * blockDim.x + threadIdx.x] = W3[0][element][i][channel_id];

		if (i < w2_size)
		{
			buffer_W2[i * blockDim.x + threadIdx.x] = W2[0][element][i][channel_id];
		}

		if (i < w1_size)
		{
			buffer_W1[i * blockDim.x + threadIdx.x] = W1[0][element][i][channel_id];
		}

		if (i < nl)
		{
			buffer_X[i * blockDim.x + threadIdx.x] = X[blockIdx.x][i][channel_id];

			for (int j = threadIdx.x; j < nl; j += blockDim.x)
			{
				int num_nonzero_u3 = U3_num_nonzero[0][i][j];

				buffer_u3_nonzeros[i * nl + j] = num_nonzero_u3;

				for (int k = 0; k < num_nonzero_u3; k++)
				{
					buffer_u3_indices[i * (nl * u3_maxn_nonsparse) + (k * nl) + j] = U3_indices[k][i][j]; // packed 32 bit integer containing 4 x uint8 indices
					buffer_u3_values[i * (nl * u3_maxn_nonsparse) + (k * nl) + j] = U3_values[k][i][j];
				}

				buffer_u2_nonzero[i * nl + j] = U2_num_nonzero[0][i][j];
				buffer_u2_indices[i * nl + j] = U2_indices[0][i][j];
				buffer_u2_values[i * nl + j] = U2_values[0][i][j];
			}
		}
	}

	buffer_out[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;

	__syncthreads();

	scalar_t output_1 = 0.0;

	for (int i = threadIdx.y; i < nl; i += blockDim.y)
	{
		scalar_t Xi = buffer_X[i * blockDim.x + threadIdx.x];

		scalar_t uw1 = 0.0;

		if (i == 0)
		{
			uw1 = buffer_W1[0 * blockDim.x + threadIdx.x];
		}

		scalar_t output_2 = 0.0;
		scalar_t deriv1_tmp = 0.0;

		if (requires_grad)
			deriv1_tmp = uw1;

		for (int j = 0; j < nl; j++)
		{
			scalar_t Xj = buffer_X[j * blockDim.x + threadIdx.x];

			scalar_t uw2 = 0.0;

			if (buffer_u2_nonzero[i * nl + j] > 0)
			{
				uw2 = buffer_u2_values[i * nl + j] * buffer_W2[buffer_u2_indices[i * nl + j] * blockDim.x + threadIdx.x];
			}

			// int uw3_num_nonzero = buffer_u3_nonzeros[i * nl + j];

			scalar_t output_3 = 0.0;
			scalar_t deriv_1_j_tmp = 0.0;

			if (requires_grad)
				deriv_1_j_tmp = uw2;

			for (int k = 0; k < buffer_u3_nonzeros[i * nl + j]; k++)
			{
				volatile int u3_mem_idx = i * (nl * u3_maxn_nonsparse) + (k * nl) + j;

				int compressed_indices = buffer_u3_indices[u3_mem_idx];

				int u3_ldx1 = compressed_indices & 0xFF;
				int u3_kdx = (compressed_indices >> 8) & 0xFF;

				scalar_t w3_1 = buffer_W3[u3_ldx1 * blockDim.x + threadIdx.x];

				scalar_t u3 = buffer_u3_values[u3_mem_idx];

				scalar_t Xk = buffer_X[u3_kdx * blockDim.x + threadIdx.x];

				output_3 += u3 * w3_1 * Xk;

				if (requires_grad)
				{
					// int u3_ldx3 = ;
					// int u3_ldx2 =;

					// scalar_t w3_2 = ;
					// scalar_t w3_3 = ;

					deriv_1_j_tmp += u3 * (w3_1 + buffer_W3[((compressed_indices >> 24) & 0xFF) * blockDim.x + threadIdx.x] + buffer_W3[((compressed_indices >> 16) & 0xFF) * blockDim.x + threadIdx.x]) * Xk;
				}
			}
			output_2 += (output_3 + uw2) * Xj;
			deriv1_tmp += (uw2 + deriv_1_j_tmp) * Xj;
		}

		output_1 += (output_2 + uw1) * Xi;

		if (requires_grad)
		{
			grad[blockIdx.x][0][i][channel_id] = deriv1_tmp;
		}

		buffer_out[threadIdx.y * blockDim.x + threadIdx.x] = output_1;

		__syncthreads();

		if (threadIdx.y == 0)
		{
			scalar_t output = 0.0;

			for (int i = 0; i < blockDim.y; i++)
			{
				output += buffer_out[i * blockDim.x + threadIdx.x];
			}

			out[blockIdx.x][0][channel_id] = output;
		}
	}
}

std::vector<torch::Tensor> symmetric_contraction_L0_forwards_gpu(
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor U3_num_nonzero,
	torch::Tensor U3_indices,
	torch::Tensor U3_values,
	torch::Tensor U2_num_nonzero,
	torch::Tensor U2_indices,
	torch::Tensor U2_values,
	torch::Tensor U1_num_nonzero,
	torch::Tensor U1_indices,
	torch::Tensor W3,
	torch::Tensor W2,
	torch::Tensor W1,
	const int w3_size,
	const int w2_size,
	const int w1_size,
	torch::Tensor u3_n_nonsparse,
	const int64_t nthreadX = 32,
	const int64_t nthreadY = 4,
	const int64_t nthreadZ = 1)
{

	torch::Tensor output = torch::zeros({X.size(0), 1, X.size(2)},
										torch::TensorOptions()
											.dtype(X.dtype())
											.device(X.device()));
	torch::Tensor grad;

	if (X.requires_grad())
	{
		grad = torch::zeros({X.size(0), 1, X.size(1), X.size(2)},
							torch::TensorOptions()
								.dtype(X.dtype())
								.device(X.device()));
	}
	else
	{
		grad = torch::zeros({1, 1, 1, 1},
							torch::TensorOptions()
								.dtype(X.dtype())
								.device(X.device()));
	}

	const int natoms = X.size(0);
	const int nl = X.size(1);
	const int nchannels = X.size(2);

	auto find_num_blocks = [](int x, int bdim)
	{ return (x + bdim - 1) / bdim; };

	dim3 block_dim(natoms, nchannels / nthreadX);
	// dim3 block_dim(natoms);

	dim3 grid(nthreadX, nthreadY, nthreadZ);
	AT_DISPATCH_FLOATING_TYPES(
		X.type(), "symmetric_contraction_L0_forwards_gpu", ([&]
															{
            size_t shared_size = 0;

			void* sptr = nullptr;

			shared_array<scalar_t>(nthreadX * nl, sptr, &shared_size);
			shared_array<scalar_t>(nthreadY * nthreadX, sptr, &shared_size);
			shared_array<float>(u3_n_nonsparse[0].item<int>() * nl * nl, sptr, &shared_size);
			shared_array<float>(nl * nl, sptr, &shared_size);
			shared_array<scalar_t>(w3_size * nthreadX, sptr, &shared_size);
			shared_array<scalar_t>(w2_size * nthreadX, sptr, &shared_size);
			shared_array<scalar_t>(w1_size * nthreadX, sptr, &shared_size);

			shared_array<int>(u3_n_nonsparse[0].item<int>() * nl * nl, sptr, &shared_size);
			shared_array<short>(nl * nl, sptr, &shared_size);
			shared_array<short>(nl * nl, sptr, &shared_size);
			shared_array<short>(nl * nl, sptr, &shared_size);
			//shared_array<short>(nl, sptr, &shared_size);
			//shared_array<short>(nl, sptr, &shared_size);
            symmetric_contraction_L0_forwards_kernel<<<block_dim, grid, shared_size>>>(
				X.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
				U3_num_nonzero.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
				U3_indices.packed_accessor32<int, 3, torch::RestrictPtrTraits>(), // L=0 specific
				U3_values.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), // L=0 specific
				U2_num_nonzero.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
				U2_indices.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
				U2_values.packed_accessor32<float,3, torch::RestrictPtrTraits>(),
				U1_num_nonzero.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
				U1_indices.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
				W3.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				W2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				W1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				w3_size,
				w2_size,
				w1_size,
				u3_n_nonsparse[0].item<int>(),
				output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				X.requires_grad()); }));

	cudaDeviceSynchronize();

	return {output, grad};
}

template <typename scalar_t>
__global__ void symmetric_contraction_LGT0_forwards_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> atom_types,

	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U3_num_nonzero_1,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U3_num_nonzero_2,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U3_num_nonzero_3,

	const torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> U3_indices_1,
	const torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> U3_indices_2,
	const torch::PackedTensorAccessor32<short, 4, torch::RestrictPtrTraits> U3_indices_3,

	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U3_values_1,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U3_values_2,
	const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U3_values_3,

	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_num_nonzero_1,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_num_nonzero_2,

	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_indices_1,
	const torch::PackedTensorAccessor32<short, 3, torch::RestrictPtrTraits> U2_indices_2,

	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> U2_values_1,
	const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> U2_values_2,

	const torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> U1_num_nonzero,
	const torch::PackedTensorAccessor32<short, 2, torch::RestrictPtrTraits> U1_index,

	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W3,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W2,
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> W1,

	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> W3_size,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> W2_size,
	const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> W1_size,

	torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> out,
	torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
	const bool requires_grad)
{

	extern __shared__ char buffer[];

	const int natoms = X.size(0);
	const int nl = X.size(1);
	const int nchannels = X.size(2);
	const int nlout = U3_values_1.size(0);
	const int u3_maxn_nonsparse = U3_values_1.size(1);

	void *sptr = buffer;

	volatile scalar_t *buffer_X = shared_array<scalar_t>(blockDim.z * blockDim.x * nl, sptr);
	volatile scalar_t *buffer_out = shared_array<scalar_t>(blockDim.y * blockDim.x, sptr);

	volatile scalar_t *buffer_W3 = shared_array<scalar_t>(W3.size(2) * blockDim.x, sptr);
	volatile scalar_t *buffer_W2 = shared_array<scalar_t>(W2.size(2) * blockDim.x, sptr);
	volatile scalar_t *buffer_W1 = shared_array<scalar_t>(W1.size(2) * blockDim.x, sptr);

	volatile float *buffer_U3_values_1 = shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr);
	// volatile float *buffer_U3_values_2 = shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr);
	volatile float *buffer_U3_values_3 = shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr);

	volatile float *buffer_U2_values_1 = shared_array<float>(nl * nl, sptr);
	// volatile float *buffer_U2_values_2 = shared_array<float>(nl * nl, sptr);

	volatile short *buffer_U3_num_nonzero_1 = shared_array<short>(nl * nl, sptr);
	// volatile short *buffer_U3_num_nonzero_2 = shared_array<short>(nl * nl, sptr);
	volatile short *buffer_U3_num_nonzero_3 = shared_array<short>(nl * nl, sptr);

	volatile short *buffer_U3_indices_1 = shared_array<short>(u3_maxn_nonsparse * nl * nl, sptr);
	// volatile short *buffer_U3_indices_2 = shared_array<short>(u3_maxn_nonsparse * nl * nl, sptr);
	volatile short *buffer_U3_indices_3 = shared_array<short>(u3_maxn_nonsparse * nl * nl, sptr);

	volatile short *buffer_U2_num_nonzero_1 = shared_array<short>(nl * nl, sptr);
	// volatile short *buffer_U2_num_nonzero_2 = shared_array<short>(nl * nl, sptr);

	volatile short *buffer_U2_indices_1 = shared_array<short>(nl * nl, sptr);
	// volatile short *buffer_U2_indices_2 = shared_array<short>(nl * nl, sptr);

	int channel_id = blockIdx.y * blockDim.x + threadIdx.x;

	int lout = 1 + blockIdx.z;

	/** load X into shared memory **/

	__syncthreads();

	int w3_size = W3_size[lout];
	int w2_size = W2_size[lout];
	int w1_size = W1_size[lout];

	int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

	int x = global_idx % 16;
	int y = global_idx / 16;
	int nthreadsy = blockDim.x * blockDim.y / 16;

	if (threadIdx.z == 0)
	{
		for (int i = y; i < nl; i += nthreadsy)
		{
			int U3_nsparse_1 = U3_num_nonzero_1[lout][i][x];

			buffer_U3_num_nonzero_1[i * nl + x] = U3_nsparse_1;

			for (int k = 0; k < U3_nsparse_1; k++)
			{
				int u3_memidx = i * (nl * u3_maxn_nonsparse) + (k * nl) + x;

				buffer_U3_indices_1[u3_memidx] = U3_indices_1[lout][k][i][x];
				buffer_U3_values_1[u3_memidx] = U3_values_1[lout][k][i][x];
			}

			if (requires_grad)
			{
				int U3_nsparse_3 = U3_num_nonzero_3[lout][i][x];

				buffer_U3_num_nonzero_3[i * nl + x] = U3_nsparse_3;

				for (int k = 0; k < U3_nsparse_3; k++)
				{
					int u3_memidx = i * (nl * u3_maxn_nonsparse) + (k * nl) + x;

					buffer_U3_indices_3[u3_memidx] = U3_indices_3[lout][k][i][x];
					buffer_U3_values_3[u3_memidx] = U3_values_3[lout][k][i][x];
				}
			}

			buffer_U2_num_nonzero_1[i * nl + x] = U2_num_nonzero_1[lout][i][x];
			buffer_U2_indices_1[i * nl + x] = U2_indices_1[lout][i][x];
			buffer_U2_values_1[i * nl + x] = U2_values_1[lout][i][x];
		}
	}

	__syncthreads();

	int prev_element = -1;

	for (int ii = threadIdx.z; ii < NATOMS_PER_BLOCK; ii += blockDim.z)
	{
		int atom_id = blockIdx.x * NATOMS_PER_BLOCK + ii;

		if (atom_id >= natoms)
		{
			return;
		}

		int element = atom_types[atom_id];

		for (int i = threadIdx.y; i < nl; i += blockDim.y)
		{
			buffer_X[threadIdx.z * (nl * blockDim.x) + i * blockDim.x + threadIdx.x] = X[atom_id][i][channel_id];
		}

		if (element != prev_element)
		{
			for (int i = threadIdx.y; i < w3_size; i += blockDim.y)
			{
				buffer_W3[i * blockDim.x + threadIdx.x] = W3[lout][element][i][channel_id];
			}

			for (int i = threadIdx.y; i < w2_size; i += blockDim.y)
			{
				buffer_W2[i * blockDim.x + threadIdx.x] = W2[lout][element][i][channel_id];
			}

			for (int i = threadIdx.y; i < w1_size; i += blockDim.y)
			{
				buffer_W1[i * blockDim.x + threadIdx.x] = W1[lout][element][i][channel_id];
			}

			prev_element = element;
		}

		buffer_out[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;

		__syncthreads();

		scalar_t output_1 = 0.0;

		// TODO use Z to parallelize over this dimension, Y dimension then goes over channel_dim...
		for (int i = threadIdx.y; i < nl; i += blockDim.y)
		{
			scalar_t output2 = 0.0;
			// first compute part of the forwards that we need for the backwards...
			scalar_t dB = 0.0;

			// TODO use X to parallelize over this dimension, and warp reduce sum later....

			for (int j = 0; j < nl; j++)
			{
				// compute uw2...
				scalar_t uw2 = 0.0;

				if (buffer_U2_num_nonzero_1[i * nl + j] > 0)
				{
					uw2 = buffer_U2_values_1[i * nl + j] * buffer_W2[buffer_U2_indices_1[i * nl + j] * blockDim.x + threadIdx.x];
				}

				// compute correlation=3 contraction
				scalar_t output3 = 0.0;

				int nk3 = buffer_U3_num_nonzero_1[i * nl + j];

				for (int k = 0; k < nk3; k++)
				{
					int u3_memidx = i * (nl * u3_maxn_nonsparse) + (k * nl) + j;

					int compressed_index = buffer_U3_indices_1[u3_memidx];

					int ldx = (int)(compressed_index & 0xFF);
					int kdx = (int)((compressed_index >> 8) & 0xFF);

					scalar_t u3 = buffer_U3_values_1[u3_memidx];

					scalar_t X_kdx = buffer_X[threadIdx.z * (nl * blockDim.x) + kdx * blockDim.x + threadIdx.x];

					scalar_t w = buffer_W3[ldx * blockDim.x + threadIdx.x];

					output3 += u3 * X_kdx * w;
				}

				output2 += (output3 + uw2) * buffer_X[threadIdx.z * (nl * blockDim.x) + j * blockDim.x + threadIdx.x];
			}

			if (requires_grad)
			{
				for (int j = 0; j < nl; j++)
				{
					scalar_t Aia = 0.0;
					scalar_t dBi = 0.0;

					// compute uw2...
					scalar_t uw2 = 0.0;

					if (buffer_U2_num_nonzero_1[j * nl + i] > 0)
					{
						uw2 = buffer_U2_values_1[j * nl + i] * buffer_W2[buffer_U2_indices_1[j * nl + i] * blockDim.x + threadIdx.x];
					}

					int nk3_2 = buffer_U3_num_nonzero_1[j * nl + i];

					for (int k = 0; k < nk3_2; k++)
					{
						int u3_memidx = j * (nl * u3_maxn_nonsparse) + (k * nl) + i;

						int compressed_index = buffer_U3_indices_1[u3_memidx];

						int ldx = (int)(compressed_index & 0xFF);
						int kdx = (int)((compressed_index >> 8) & 0xFF);

						scalar_t u3 = buffer_U3_values_1[u3_memidx];

						scalar_t X_kdx = buffer_X[threadIdx.z * (nl * blockDim.x) + kdx * blockDim.x + threadIdx.x];

						scalar_t w = buffer_W3[ldx * blockDim.x + threadIdx.x];

						Aia += u3 * X_kdx * w;
					}

					int nk3_3 = buffer_U3_num_nonzero_3[i * nl + j];

					for (int k = 0; k < nk3_3; k++)
					{
						int u3_memidx = i * (nl * u3_maxn_nonsparse) + (k * nl) + j;

						int compressed_index = buffer_U3_indices_3[u3_memidx];

						int ldx = (int)(compressed_index & 0xFF);
						int kdx = (int)((compressed_index >> 8) & 0xFF);

						scalar_t u3 = buffer_U3_values_3[u3_memidx];

						scalar_t X_kdx = buffer_X[threadIdx.z * (nl * blockDim.x) + kdx * blockDim.x + threadIdx.x];

						scalar_t w = buffer_W3[ldx * blockDim.x + threadIdx.x];

						dBi += u3 * X_kdx * w;
					}
					dBi += Aia + uw2;

					dB += dBi * buffer_X[threadIdx.z * (nl * blockDim.x) + j * blockDim.x + threadIdx.x];
				}
			}

			scalar_t uw1 = 0.0;

			if (U1_num_nonzero[lout][i] > 0)
			{
				int uw1_idx = U1_index[lout][i];

				uw1 = buffer_W1[uw1_idx * blockDim.x + threadIdx.x];
			}

			output_1 += (output2 + uw1) * buffer_X[threadIdx.z * (nl * blockDim.x) + i * blockDim.x + threadIdx.x];

			if (requires_grad)
			{

				// reduce over channel_dim with warp instructions

				/*


				__syncwarp();

				if (threadIdx.x == 0) {
					grad[atom_id][lout][threadIdx.z -> i][threadIdx.y -> channel_id] = dB;
					}
				}*/

				dB += output2 + uw1;
				grad[atom_id][lout][i][channel_id] = dB;
			}
		}

		// reduce over channel_dim with warp instructions

		atomicAdd((scalar_t *)&buffer_out[threadIdx.y * blockDim.x + threadIdx.x], output_1);

		__syncthreads();

		if (threadIdx.y == 0)
		{
			scalar_t output_sum = 0.0;

			for (int i = 0; i < blockDim.y; i++)
			{
				output_sum += buffer_out[i * blockDim.x + threadIdx.x];
			}

			out[atom_id][lout][channel_id] = output_sum;
		}
	}
}

std::vector<torch::Tensor> symmetric_contraction_LGT0_forwards_gpu(
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
	const int64_t W3_l0_size,
	const int64_t W2_l0_size,
	const int64_t W1_l0_size,
	torch::Tensor W3_size,			// nlout
	torch::Tensor W2_size,			// nlout
	torch::Tensor W1_size,			// nlout
	torch::Tensor U3_max_nonsparse, // nlout
	const int64_t nthreadX = 32,
	const int64_t nthreadY = 4,
	const int64_t nthreadZ = 1)
{

	const int natoms = X.size(0);
	const int nl = X.size(1);
	const int nchannels = X.size(2);
	const int nlout = U3_num_nonzero_1.size(0);
	const int u3_maxn_nonsparse = U3_values_1.size(1);

	auto find_num_blocks = [](int x, int bdim)
	{ return (x + bdim - 1) / bdim; };

	torch::Tensor output = torch::zeros({natoms, nlout, nchannels},
										torch::TensorOptions()
											.dtype(X.dtype())
											.device(X.device()));

	torch::Tensor grad;

	if (X.requires_grad())
	{
		grad = torch::zeros({natoms, nlout, nl, nchannels},
							torch::TensorOptions()
								.dtype(X.dtype())
								.device(X.device()));
	}
	else
	{
		grad = torch::zeros({1, 1, 1, 1},
							torch::TensorOptions()
								.dtype(X.dtype())
								.device(X.device()));
	}

	dim3 gridDim_0(natoms, nchannels / nthreadX);
	dim3 blockDim_0(nthreadX, nthreadY, 1);

	dim3 gridDim(find_num_blocks(natoms, NATOMS_PER_BLOCK), nchannels / nthreadX, nlout - 1);
	dim3 blockDim(nthreadX, nthreadY, nthreadZ);

	AT_DISPATCH_FLOATING_TYPES(
		X.type(), "symmetric_contraction_LGT0_forwards_gpu", ([&]
															  {

		size_t shared_size = 0;
		void* sptr = nullptr;

		shared_array<scalar_t>(nthreadX * nl, sptr, &shared_size);
		shared_array<scalar_t>(nthreadY * nthreadX, sptr, &shared_size);
		shared_array<float>(U3_max_nonsparse[0].item<int>()* nl * nl, sptr, &shared_size);
		shared_array<float>(nl * nl, sptr, &shared_size);
		shared_array<scalar_t>(W3_l0_size * nthreadX, sptr, &shared_size);
		shared_array<scalar_t>(W2_l0_size * nthreadX, sptr, &shared_size);
		shared_array<scalar_t>(W1_l0_size * nthreadX, sptr, &shared_size);

		shared_array<int>(U3_max_nonsparse[0].item<int>() * nl * nl, sptr, &shared_size);
		shared_array<short>(nl * nl, sptr, &shared_size);
		shared_array<short>(nl * nl, sptr, &shared_size);
		shared_array<short>(nl * nl, sptr, &shared_size);
		shared_array<short>(nl, sptr, &shared_size);
		shared_array<short>(nl, sptr, &shared_size);

		//printf("check L=0: %d\n", shared_size);

		symmetric_contraction_L0_forwards_kernel<scalar_t><<<gridDim_0, blockDim_0, shared_size>>>(
			X.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
			atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			U3_num_nonzero_1.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
			U3_indices_0.packed_accessor32<int, 3, torch::RestrictPtrTraits>(), // L = 0 specific
			U3_values_0.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), // L = 0 specific
			U2_num_nonzero_1.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
			U2_indices_1.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
			U2_values_1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			U1_num_nonzero.packed_accessor32<short,2, torch::RestrictPtrTraits>(),
			U1_index.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
			W3.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			W2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			W1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			W3_l0_size,
			W2_l0_size,
			W1_l0_size,
			U3_max_nonsparse[0].item<int>(),
			output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
			grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
			X.requires_grad());

	shared_size = 0;
	sptr = nullptr;

	shared_array<scalar_t>(blockDim.z * blockDim.x * nl, sptr, &shared_size);
	shared_array<scalar_t>(blockDim.x * blockDim.y, sptr, &shared_size);

	shared_array<scalar_t>(W3.size(2) * blockDim.x, sptr, &shared_size);
	shared_array<scalar_t>(W2.size(2) * blockDim.x, sptr, &shared_size);
	shared_array<scalar_t>(W1.size(2) * blockDim.x, sptr, &shared_size);

	shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr, &shared_size);
	//shared_array<float>(u3_maxn_nonsparse * blockDim.y * nl, sptr, &shared_size);
	shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr, &shared_size);

	shared_array<float>(nl * nl, sptr, &shared_size);
	//shared_array<float>(nl * nl, sptr, &shared_size);

	shared_array<short>(nl * nl, sptr, &shared_size);
	//shared_array<short>(blockDim.y * nl, sptr, &shared_size);
	shared_array<short>(nl * nl, sptr, &shared_size);

	shared_array<short>(u3_maxn_nonsparse *nl * nl, sptr, &shared_size);
	//shared_array<short>(u3_maxn_nonsparse * blockDim.y * nl, sptr, &shared_size);
	shared_array<short>(u3_maxn_nonsparse *nl * nl, sptr, &shared_size);

	shared_array<short>(nl * nl, sptr, &shared_size);
	//shared_array<short>(blockDim.y * nl, sptr, &shared_size);

	shared_array<short>(nl * nl, sptr, &shared_size);
	//shared_array<short>(blockDim.y * nl, sptr, &shared_size);

	//printf("check L>0: %d\n", shared_size);

	symmetric_contraction_LGT0_forwards_kernel<scalar_t><<<gridDim, blockDim, shared_size>>>(
		X.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		U3_num_nonzero_1.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
		U3_num_nonzero_2.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
		U3_num_nonzero_3.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),

		U3_indices_1.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
		U3_indices_2.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),
		U3_indices_3.packed_accessor32<short, 4, torch::RestrictPtrTraits>(),

		U3_values_1.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		U3_values_2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		U3_values_3.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),

		U2_num_nonzero_1.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
		U2_num_nonzero_2.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),

		U2_indices_1.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),
		U2_indices_2.packed_accessor32<short, 3, torch::RestrictPtrTraits>(),

		U2_values_1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		U2_values_2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),

		U1_num_nonzero.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),
		U1_index.packed_accessor32<short, 2, torch::RestrictPtrTraits>(),

		W3.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
		W2.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
		W1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),

		W3_size.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		W2_size.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		W1_size.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),

		output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
		grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
		X.requires_grad()); }));

	cudaDeviceSynchronize();

	return {output, grad};
}

template <typename scalar_t>
__global__ void symm_contraction_backward_kernel(
	const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_X,
	const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_input,
	torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output)
{
	extern __shared__ char buffer[];

	const int natoms = grad_X.size(0);
	const int nlout = grad_X.size(1);
	const int nl = grad_X.size(2);
	const int nchannels = grad_X.size(3);

	size_t offset = 0;

	scalar_t *buffer_grad = reinterpret_cast<scalar_t *>(buffer + offset);
	offset += blockDim.x * nl * sizeof(scalar_t);

	int atom_idx = blockIdx.x;
	int channel_id = blockIdx.y * blockDim.x + threadIdx.x;

	for (int sph = threadIdx.y; sph < nl; sph += blockDim.y)
	{
		buffer_grad[sph * blockDim.x + threadIdx.x] = 0.0;
	}

	__syncthreads();

	for (int lout = 0; lout < nlout; lout++)
	{
		scalar_t grad = grad_input[atom_idx][lout][channel_id];

		for (int sph = threadIdx.y; sph < nl; sph += blockDim.y)
		{
			buffer_grad[sph * blockDim.x + threadIdx.x] += grad * grad_X[atom_idx][lout][sph][channel_id];
		}
	}

	__syncthreads();

	for (int sph = threadIdx.y; sph < nl; sph += blockDim.y)
	{
		grad_output[atom_idx][sph][channel_id] = buffer_grad[sph * blockDim.x + threadIdx.x];
	}
}

torch::Tensor symm_contraction_backward(
	torch::Tensor gradX,
	torch::Tensor grad_input,
	int nthreadX = 32,
	int nthreadY = 4,
	int nthreadZ = 1)
{

	const int natoms = gradX.size(0);
	const int nlout = gradX.size(1);
	const int nl = gradX.size(2);
	const int nchannels = gradX.size(3);

	torch::Tensor output = torch::zeros({natoms, nl, nchannels},
										torch::TensorOptions()
											.dtype(gradX.dtype())
											.device(gradX.device()));

	dim3 block_dim(natoms, nchannels / nthreadX);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	AT_DISPATCH_FLOATING_TYPES(
		gradX.type(), "symm_contraction_backward", ([&]
													{

			size_t shared_mem_amount =  nthreadX * nl * sizeof(scalar_t); // buffer_grad storage

			symm_contraction_backward_kernel<<<block_dim, grid, shared_mem_amount>>>(
				gradX.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
				grad_input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
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
		const int64_t nthreadz)
	{
		const int nlout = U3_values_1.size(0);

		std::vector<torch::Tensor> result;

		if (nlout == 1)
		{
			// use special l=0 kernel
			result = symmetric_contraction_L0_forwards_gpu(
				X,
				atom_types,
				U3_num_nonzero_1,
				U3_indices_0,
				U3_values_0,
				U2_num_nonzero_1,
				U2_indices_1,
				U2_values_1,
				U1_num_nonzero,
				U1_index,
				W3,
				W2,
				W1,
				W3_L0_size,
				W2_L0_size,
				W1_L0_size,
				U3_max_nonsparse,
				nthreadx,
				nthready,
				nthreadz);

			if (X.requires_grad())
			{
				ctx->save_for_backward({result[1]});
			}
		}
		else
		{
			// use generic kernel
			result = symmetric_contraction_LGT0_forwards_gpu(
				X,
				atom_types,
				U3_num_nonzero_1,
				U3_num_nonzero_2,
				U3_num_nonzero_3,
				U3_indices_0,
				U3_indices_1,
				U3_indices_2,
				U3_indices_3,
				U3_values_0,
				U3_values_1,
				U3_values_2,
				U3_values_3,
				U2_num_nonzero_1,
				U2_num_nonzero_2,
				U2_indices_1,
				U2_indices_2,
				U2_values_1,
				U2_values_2,
				U1_num_nonzero,
				U1_index,
				W3,
				W2,
				W1,
				W3_L0_size,
				W2_L0_size,
				W1_L0_size,
				W3_size,
				W2_size,
				W1_size,
				U3_max_nonsparse,
				nthreadx,
				nthready,
				nthreadz);

			if (X.requires_grad())
			{
				ctx->save_for_backward({result[1]});
			}
		}

		if (X.requires_grad())
		{
			ctx->saved_data["nthreadx"] = nthreadx;
			ctx->saved_data["nthready"] = nthready;
			ctx->saved_data["nthreadz"] = nthreadz;
		}

		return result[0];
	}

	static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
	{
		auto saved_variables = ctx->get_saved_variables();

		int nthreadx = ctx->saved_data["nthreadx"].toInt();
		int nthready = ctx->saved_data["nthready"].toInt();
		int nthreadz = ctx->saved_data["nthreadz"].toInt();

		auto gradX = saved_variables[0];

		torch::Tensor result = symm_contraction_backward(gradX, grad_outputs[0], nthreadx, nthready, nthreadz);

		torch::Tensor undef;

		return {result,
				undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef,
				undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef,
				undef};
	}
};

torch::Tensor symmetric_contraction(
	torch::Tensor X,
	torch::Tensor atom_types,
	torch::Tensor U3_num_nonzero_1,
	torch::Tensor U3_num_nonzero_2,
	torch::Tensor U3_num_nonzero_3,
	torch::Tensor U3_indices_0, // L=0 specific
	torch::Tensor U3_indices_1,
	torch::Tensor U3_indices_2,
	torch::Tensor U3_indices_3,
	torch::Tensor U3_values_0, // L=0 specific
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
	torch::Tensor W3_size,
	torch::Tensor W2_size,
	torch::Tensor W1_size,
	torch::Tensor U3_max_nonsparse,
	const int64_t nthreadx,
	const int64_t nthready,
	const int64_t nthreadz)
{

	return SymmetricContractionAutograd::apply(
		X,
		atom_types,
		U3_num_nonzero_1,
		U3_num_nonzero_2,
		U3_num_nonzero_3,
		U3_indices_0,
		U3_indices_1,
		U3_indices_2,
		U3_indices_3,
		U3_values_0,
		U3_values_1,
		U3_values_2,
		U3_values_3,
		U2_num_nonzero_1,
		U2_num_nonzero_2,
		U2_indices_1,
		U2_indices_2,
		U2_values_1,
		U2_values_2,
		U1_num_nonzero,
		U1_index,
		W3,
		W2,
		W1,
		W3_L0_size,
		W2_L0_size,
		W1_L0_size,
		W3_size,
		W2_size,
		W1_size,
		U3_max_nonsparse,
		nthreadx,
		nthready,
		nthreadz);
}

int64_t curr_shared_mem(int64_t device)
{

	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, device);

	return deviceProp.sharedMemPerBlock;
}

int64_t LGT0_shared_memory_required(int64_t nthreadsX, int64_t nthreadsY, int64_t nthreadsZ, int64_t u3_maxn_nonsparse, int64_t nl, int64_t W3_size, int64_t W2_size, int64_t W1_size, torch::ScalarType scalar_type)
{
	size_t shared_size = 0;
	void *sptr = nullptr;

	switch (scalar_type)
	{
	case torch::ScalarType::Double:
		shared_array<double>(nthreadsZ * nthreadsX * nl, sptr, &shared_size);
		shared_array<double>(nthreadsX * nthreadsY, sptr, &shared_size);

		shared_array<double>(W3_size * nthreadsX, sptr, &shared_size);
		shared_array<double>(W2_size * nthreadsX, sptr, &shared_size);
		shared_array<double>(W1_size * nthreadsX, sptr, &shared_size);
		break;
	case torch::ScalarType::Float:
		shared_array<float>(nthreadsZ * nthreadsX * nl, sptr, &shared_size);
		shared_array<float>(nthreadsX * nthreadsY, sptr, &shared_size);

		shared_array<float>(W3_size * nthreadsX, sptr, &shared_size);
		shared_array<float>(W2_size * nthreadsX, sptr, &shared_size);
		shared_array<float>(W1_size * nthreadsX, sptr, &shared_size);
		break;
	}

	shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr, &shared_size);
	// shared_array<float>(u3_maxn_nonsparse * nthreadsY * nl, sptr, &shared_size);
	shared_array<float>(u3_maxn_nonsparse * nl * nl, sptr, &shared_size);

	shared_array<float>(nl * nl, sptr, &shared_size);
	// shared_array<float>(nthreadsY * nl, sptr, &shared_size);

	shared_array<short>(nl * nl, sptr, &shared_size);
	// shared_array<short>(nthreadsY * nl, sptr, &shared_size);
	shared_array<short>(nl * nl, sptr, &shared_size);

	shared_array<short>(u3_maxn_nonsparse * nl * nl, sptr, &shared_size);
	// shared_array<short>(u3_maxn_nonsparse * nthreadsY * nl, sptr, &shared_size);
	shared_array<short>(u3_maxn_nonsparse * nl * nl, sptr, &shared_size);

	shared_array<short>(nl * nl, sptr, &shared_size);
	// shared_array<short>(nthreadsY * nl, sptr, &shared_size);

	shared_array<short>(nl * nl, sptr, &shared_size);
	// shared_array<short>(nthreadsY * nl, sptr, &shared_size);

	return shared_size;
}

bool set_shared_mem_size(int64_t amount, int64_t device, torch::ScalarType scalar_type)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	size_t dtype = torch::elementSize(scalar_type);

	bool accepted = amount <= deviceProp.sharedMemPerBlockOptin;

	if (!accepted)
	{
		std::cerr << "Warning: requested shared memory buffer (" << amount;
		std::cerr << ") exceeds max available (" << deviceProp.sharedMemPerBlockOptin;
		std::cerr << ") on device " << deviceProp.name << std::endl;
	}
	else
	{
		switch (scalar_type)
		{
		case torch::ScalarType::Double:
			cudaFuncSetAttribute(
				symmetric_contraction_LGT0_forwards_kernel<double>,
				cudaFuncAttributeMaxDynamicSharedMemorySize,
				amount);

			break;
		case torch::ScalarType::Float:
			cudaFuncSetAttribute(
				symmetric_contraction_LGT0_forwards_kernel<float>,
				cudaFuncAttributeMaxDynamicSharedMemorySize,
				amount);
			break;
		}
	}

	return accepted;
}

TORCH_LIBRARY(mace_cuda_symm_contraction, m)
{
	m.def("symmetric_contraction", &symmetric_contraction);
	m.def("set_shared_mem_size", &set_shared_mem_size);
	m.def("curr_shared_mem", &curr_shared_mem);
	m.def("LGT0_shared_memory_required", &LGT0_shared_memory_required);
}
