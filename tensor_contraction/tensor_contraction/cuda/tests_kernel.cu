#include <math.h>
#include<torch/torch.h>
#include <iostream>
#include <mma.h>

using namespace std;
using namespace nvcuda;

#define NWARPS 6
#define WARP_SIZE_Y 4
#define WARP_SIZE_Z 8

__global__ void wmma_dense_tensorcore_kernel_16x16_f32(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U,
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

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	for (int x = 0; x < 48; x += 16) { // 3

		for (int y = 0; y < 96; y += 16) { // 6

			//load 16x16 chunks
			for (int i = tidx; i < 16; i += blockDim.x) { // m, n

				for (int k = tidy; k < 16; k += blockDim.y) { // k

					su[i][k] = __float2half(U[blockIdx.x][blockIdx.y][x + i][k]);

					sw[k][i] = __float2half(weights[k][y + i]);
				}
			}

			__syncwarp();

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

void Uw_dense_tensorcore(torch::Tensor U, torch::Tensor W, torch::Tensor C) {

	const int nthreadsx = 4;
	const int nthreadsy = 8;

	dim3 blocks(U.size(0), U.size(1));

	dim3 grid(nthreadsx, nthreadsy);

	wmma_dense_tensorcore_kernel_16x16_f32<<<blocks, grid, 256*4 + 256*4 >>>(
			U.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			W.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

__global__ void UwN_sparse_contraction_kernel(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Uw_dense,
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> Uw_indexes,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> Uw_nvals,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> features, torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

// Uw Shape: torch.Size([48, 48, 48, 96]) -> torch.Size([48, 48, 96, 48]) vals -> torch.Size([48, 48, 96, 6])
	// torch.Size([48, 48, 6]) indexes, penultimate dim contains the dense values

//node features: natoms, nfeatures, nirreps -> natoms, nfeatures, 6

	for (int a = blockIdx.x; a < features.size(0); a += gridDim.x) { // natoms

		for (int i = blockIdx.y; i < 48; i += gridDim.y) { // nirreps

			for (int j = threadIdx.x; j < 48; j += blockDim.x) { // nirreps

				int nval = Uw_nvals[i][j];

				if (nval == 0) {
					continue;
				}

				for (int l = threadIdx.y; l < 96; l += blockDim.y) {

					float dot = 0.0;

					for (int k = 0; k < nval; k++) {
						int idx = Uw_indexes[i][k][j];
						dot += Uw_dense[i][j][idx][l] * features[a][idx][l];
					}

					C[a][i][j][l] = dot;
				}
			}
		}
	}
}

void UwN3_sparse_contraction(torch::Tensor Uw_dense, torch::Tensor Uw_indexes, torch::Tensor Uw_nvals, torch::Tensor features, torch::Tensor C,
		const int nthreadsx, const int nthreadsy, const int nblocksx, const int nblocksy) {

	dim3 blocks(nblocksx, nblocksy);

	dim3 grid(nthreadsx, nthreadsy);

	UwN_sparse_contraction_kernel<<<blocks, grid>>>(
			Uw_dense.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			Uw_indexes.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			Uw_nvals.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			features.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}
