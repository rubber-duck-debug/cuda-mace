#include <math.h>
#include<torch/torch.h>
#include <iostream>
#include <mma.h>

using namespace std;
using namespace nvcuda;

#define WARP_SIZE 32
#define M 16
#define N 16
#define K 16

//'...ik,kc,bci -> bc...'
// 1: '...ik, kc' -> ...ic'

__global__ void UW_contraction_kernel_fp32(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights, torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

	const int nrows = U.size(2);
	const int ncols = weights.size(1);
	const int kdim = U.size(3);
//

	for (int x = blockIdx.x; x < U.size(0); x += gridDim.x) {
		for (int y = blockIdx.y; y < U.size(1); y += gridDim.y) {

			for (int i = threadIdx.x; i < C.size(2); i += blockDim.x) {
				for (int j = threadIdx.y; j < C.size(3); j += blockDim.y) {

					float product = 0.0;

					for (int k = 0; k < kdim; k++) {
						product += U[x][y][i][k] * weights[k][j];
					}

					C[x][y][i][j] = product;

				}
			}
		}
	}
}

__global__ void wmma_f32_tensorcore(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U,
		const torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> indexes,
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> nvals,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights, torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

	wmma::fragment < wmma::matrix_a, 16, 16, 16, half, wmma::row_major > a_frag;
	wmma::fragment < wmma::matrix_b, 16, 16, 16, half, wmma::row_major > b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	__shared__ half
	su[16][16]; // 48x16  -> 16x16
	__shared__ half
	sw[16][16]; // 16*96 ->  16x16

	__shared__
	float so[16][16];  // 16x16

	__shared__
	int sidx[16][16];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	for (int x = 0; x < 48; x += 16) {

		for (int y = 0; y < 96; y += 16) {

			//load 16x16 chunks
			for (int i = tidx; i < 16; i += blockDim.x) { // m, n

				for (int k = tidy; k < 16; k += blockDim.y) {

					su[i][k] = __float2half(U[blockIdx.x][blockIdx.y][x + i][k]);

					int kidx = indexes[blockIdx.x][blockIdx.y][x + i][k];

					sw[k][i] = __float2half(weights[kidx][y + i]);
				}

			}

			__syncthreads();

			// Initialize the output to zero
			wmma::fill_fragment(c_frag, 0.0f);

			wmma::load_matrix_sync(a_frag, &su[0][0], 16);
			wmma::load_matrix_sync(b_frag, &sw[0][0], 16);

			// Perform the matrix multiplication
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

			wmma::store_matrix_sync(&so[0][0], c_frag, 16, wmma::mem_row_major);

			//copy so to global memory

			for (int i = tidx; i < 16; i += blockDim.x) {
				for (int j = tidy; j < 16; j += blockDim.y) {
					C[blockIdx.x][blockIdx.y][x + i][y + j] = so[i][j];
				}
			}
		}
	}
}

__global__ void UW_contraction_kernel_sparse_fp32(const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> nnon_zero,
		const torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> indices,
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
		torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output) {

	for (int x = blockIdx.x; x < U.size(0); x += gridDim.x) {
		for (int y = blockIdx.y; y < U.size(1); y += gridDim.y) {

			for (int i = threadIdx.x; i < U.size(2); i += blockDim.x) {

				for (int j = threadIdx.y; j < output.size(3); j += blockDim.y) {

					float product = 0.0;

					int kdim = nnon_zero[x][y][i];

					for (int k = 0; k < kdim; k++) {

						int kidx = indices[x][y][i][k];

						product += U[x][y][i][kidx] * weights[kidx][j];
					}

					output[x][y][i][j] = product;

				}
			}
		}
	}
}

/**
 #                     i         j        k          l
 Uw = np.random.rand(nirreps, nirreps, nirreps, nfeatures)
 #                                 b         l         k
 node_feautures = np.random.rand(natoms, nfeatures, nirreps)

 test2 = np.zeros((natoms, nfeatures, nirreps, nirreps))
 test1 = np.einsum('...kl, blk-> bl...', Uw, node_feautures)

 for b in range(natoms):
 for c in range(nfeatures):
 for i in range(nirreps):
 v = node_feautures[b, c , i] * Uw[:,:, i, c]
 test2[b, c] += v

 scheme 1:
 blockIdx.x : b
 blockIdx.y : c
 threadIdx.x: i
 threadIdx.y: Uw[:, j]

 scheme 2:
 blockIdx.x : Uw[i, :]
 blockIdx.y : Uw[i, j, :]
 threadIdx.x: Uw[i,j, k, :]
 threadIdx.y: Uw[i,j, k, l]

 loop over natoms to construct contraction, storing local variable for output[a, ]

 **/

/*__global__ void UWn_contraction_kernel_dense_fp32(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Uw,
 const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> node_features,
 torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output) {

 for (int i = blockIdx.x; x < U.size(0); i += gridDim.x) {
 for (int j = blockIdx.y; j < U.size(1); j += gridDim.y) {

 for (int k = threadIdx.x; k < U.size(2); k += blockDim.x) {

 for (int l = threadIdx.y; l < U.size(3); l += blockDim.y) {

 for (int b = 0; k < node_features.size(0); b++) {
 output[b][c][y][k] = node_features[b][c][i] * Uw[i][j][k][l];
 }
 }
 }
 }
 }
 } */

__global__ void UW_contraction_kernel_sparse_fp64(const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> nnon_zero,
		const torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> indices,
		const torch::PackedTensorAccessor32<double, 4, torch::RestrictPtrTraits> U,
		const torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> weights,
		torch::PackedTensorAccessor32<double, 4, torch::RestrictPtrTraits> output) {

	for (int x = blockIdx.x; x < U.size(0); x += gridDim.x) {

		for (int y = blockIdx.y; y < U.size(1); y += gridDim.y) {

			for (int i = threadIdx.x; i < U.size(2); i += blockDim.x) {

				for (int j = threadIdx.y; j < output.size(3); j += blockDim.y) {

					double product = 0.0;

					int kdim = nnon_zero[x][y][i];

					for (int k = 0; k < kdim; k++) {

						int kidx = indices[x][y][i][k];

						product += U[x][y][i][kidx] * weights[kidx][j];
					}

					output[x][y][i][j] = product;

				}
			}
		}
	}
}

__global__ void UW_contraction_kernel_fp64(const torch::PackedTensorAccessor32<double, 4, torch::RestrictPtrTraits> U,
		const torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> weights,
		torch::PackedTensorAccessor32<double, 4, torch::RestrictPtrTraits> C) {

	const int nrows = U.size(2);
	const int ncols = weights.size(1);
	const int kdim = U.size(3);

	for (int i = threadIdx.x; i < C.size(2); i += blockDim.x) {
		for (int j = threadIdx.y; j < C.size(3); j += blockDim.y) {

			if (i < nrows && j < ncols) {

				double product = 0.0;

				for (int k = 0; k < kdim; k++) {
					product += U[blockIdx.x][blockIdx.y][i][k] * weights[k][j];
				}

				C[blockIdx.x][blockIdx.y][i][j] = product;
			}
		}
	}
}

// 2: '...ic, bci' -> bc...'

__global__
void UWN_contraction_kernel(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Uw,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> n, torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

}

// 1: '...ik, kc' -> ...ic'
void SparseTensor3Contraction_fp32(torch::Tensor nnon_zero, torch::Tensor indices, torch::Tensor U, torch::Tensor W, torch::Tensor C) {

	const int nthreadsx = 16;
	const int nthreadsy = 16;

	dim3 blocks(U.size(0), U.size(1));

	dim3 grid(nthreadsx, nthreadsy);

	UW_contraction_kernel_sparse_fp32<<<blocks, grid>>>(
			nnon_zero.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			indices.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
			U.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			W.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

// 1: '...ik, kc' -> ...ic'
void SparseTensor3Contraction_fp64(torch::Tensor nnon_zero, torch::Tensor indices, torch::Tensor U, torch::Tensor W, torch::Tensor C) {

	const int nthreadsx = 16;
	const int nthreadsy = 16;

	dim3 blocks(U.size(0), U.size(1));

	dim3 grid(nthreadsx, nthreadsy);

	UW_contraction_kernel_sparse_fp64<<<blocks, grid>>>(
			nnon_zero.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			indices.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
			U.packed_accessor32<double, 4, torch::RestrictPtrTraits>(),
			W.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<double, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

// 1: '...ik, kc' -> ...ic'
void WMMATensor3Contraction_fp32(torch::Tensor U, torch::Tensor indexes, torch::Tensor nvals, torch::Tensor W, torch::Tensor C) {

	const int nthreadsx = 8;
	const int nthreadsy = 8;

	dim3 blocks(U.size(0), U.size(1));

	dim3 grid(nthreadsx, nthreadsy);

	wmma_f32_tensorcore<<<blocks, grid, 256*4 + 256*4 * 2>>>(
			U.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			indexes.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
			nvals.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			W.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

// 1: '...ik, kc' -> ...ic'
void DenseTensor3Contraction_fp32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

	const int nthreadsx = 16;
	const int nthreadsy = 16;

	dim3 blocks(A.size(0), A.size(1));

	dim3 grid(nthreadsx, nthreadsy);

	UW_contraction_kernel_fp32<<<blocks, grid>>>(
			A.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			B.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void DenseTensor3Contraction_fp64(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

	const int nthreadsx = 16;
	const int nthreadsy = 16;
	dim3 blocks(A.size(0), A.size(1));

	dim3 grid(nthreadsx, nthreadsy);

	UW_contraction_kernel_fp64<<<blocks, grid>>>(
			A.packed_accessor32<double, 4, torch::RestrictPtrTraits>(),
			B.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<double, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

