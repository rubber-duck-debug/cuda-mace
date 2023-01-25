#include <math.h>
#include<torch/torch.h>
#include <iostream>
#include <mma.h>

using namespace std;
using namespace nvcuda;

__global__ void kernel_u4w_matmul_tc16x16_f32(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights, torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

	/*
	 * Computes [U[i, j, :, :] \times w[:, :]], i.e, a batched matmul over the first two dimensions of U.
	 *
	 * The final two dimensions of U and W **must** be multiples of 16 - this needs to be ensured in the calling code.
	 * Zero padding should be used to enforce this.
	 *
	 * */

	wmma::fragment < wmma::matrix_a, 16, 16, 16, half, wmma::row_major > a_frag;
	wmma::fragment < wmma::matrix_b, 16, 16, 16, half, wmma::row_major > b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	__shared__ half
	su[16][16 + 1];
	__shared__ half
	sw[16][16 + 1];

	__shared__
	float so[16][16 + 1];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int bidx = blockIdx.x;

	for (int a = bidx; a < U.size(0); a += gridDim.x) {

		for (int x = 0; x < U.size(1); x += 16) {

			for (int i = tidx; i < 16; i += blockDim.x) { // m, n

				for (int k = tidy; k < 16; k += blockDim.y) { // k

					su[i][k] = __float2half(U[a][x + i][k]);
				}
			}

			wmma::load_matrix_sync(a_frag, &su[0][0], 16);

			for (int y = 0; y < weights.size(1); y += 16) {

				for (int i = tidx; i < 16; i += blockDim.x) { // m, n

					for (int k = tidy; k < 16; k += blockDim.y) { // k

						sw[k][i] = __float2half(weights[k][y + i]);
					}
				}

				__syncwarp();

				wmma::load_matrix_sync(b_frag, &sw[0][0], 16);

				// Initialize the output to zero
				wmma::fill_fragment(c_frag, 0.0f);

				// Perform the matrix multiplication
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

				wmma::store_matrix_sync(&so[0][0], c_frag, 16, wmma::mem_row_major);

				//copy so to global memory
				for (int i = tidx; i < 16; i += blockDim.x) {
					for (int j = tidy; j < 16; j += blockDim.y) {
						C[a][x + i][y + j] = so[i][j];
					}
				}
			}

		}
	}
}

void u4w_matmul_tc16x16_f32(torch::Tensor U, torch::Tensor W, torch::Tensor C) {

	/* matrix-matrix type, operation type, tensorcore layout */

	const int nthreadsx = 4;
	const int nthreadsy = 8;

	dim3 blocks(U.size(0));

	dim3 grid(nthreadsx, nthreadsy);

	kernel_u4w_matmul_tc16x16_f32<<<blocks, grid >>>(
			U.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			W.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

