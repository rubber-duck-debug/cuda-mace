#include <math.h>
#include<torch/torch.h>
#include <iostream>
#include <mma.h>

using namespace std;
using namespace nvcuda;

#define WARP_SIZE_Y 4
#define WARP_SIZE_Z 8

__global__ void Uw3_wmma_dense_contraction_tensorcore_kernel_16x16_f32(
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
		torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

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

					su[i][k] = __float2half(
							U[blockIdx.x][blockIdx.y][x + i][k]);

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

void Uw3_dense_contraction_tensorcore(torch::Tensor U, torch::Tensor W,
		torch::Tensor C) {

	const int nthreadsx = 4;
	const int nthreadsy = 8;

	dim3 blocks(U.size(0), U.size(1));

	dim3 grid(nthreadsx, nthreadsy);

	Uw3_wmma_dense_contraction_tensorcore_kernel_16x16_f32<<<blocks, grid, 256*4 + 256*4 >>>(
			U.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			W.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

__global__ void Uw3_wmma_sparse_contraction_tensorcore_kernel_16x16_f32(
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U_values,
		const torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> U_indices,
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> U_nvals,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> Uw_nvals,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
		torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

	/***
	 *
	 * computes the matrix product U[i,j, :, :] \times w[:, :] using WMMA library. Only a single warp is used currently, which moves in a 16x16 pattern
	 * over the matrices. Possible to extend this to multiple warps which handle different 16x16 tiles.
	 *
	 * U_values are the zero-padded non-sparse values of U with shape [48,48,48,16], with corresponding U_indices: [48,48,48,16]
	 *
	 * U_values actually stored as [48,48,16,48] as this allows for better contiguous memory access across threads.
	 *
	 * Exploits sparsity in U[i,j] as only 50% of these are non-zero. Does not exploit sparsity in U[i,j,:, k] (6/48 sparsity).
	 *
	 *
	 * ***/

	//__shared__
	//float su[16][16]; // 48x16  -> 16x16
	//__shared__
	//float sw[16][16]; // 16*96 ->  16x16
	extern __shared__ float buffer[];
	float *su = &buffer[0];
	float *sw = &buffer[16 * 16];

	//int nvals = Uw_nvals[i][j] -> can skip here as U[i, j,:, :] is all zero, U[x, y, i, :] contains at most 6 non-zero vectors
	//50% of these blocks are doing matmuls on zero-valued matrices otherwise!

	for (int i = blockIdx.x; i < U_values.size(0); i += gridDim.x) {
		for (int j = blockIdx.y; j < U_values.size(1); j += gridDim.y) {

			if (Uw_nvals[i][j] == 0) {
				continue;
			}

			for (int x = 0; x < 48; x += 16) { // 3

				//sparsity over [x + m] is around 6/48, not sure if non-contiguous access of U[i,j,xx] where xx < 8 /w nthreads.x = 4 or < 4 /w nthreads.x = 2
				// would provide better performance than contiguous access of U[i,j,xx]

				for (int y = 0; y < 96; y += 16) { // 6

					/*for (int m = threadIdx.x; m < 16; m += blockDim.x) { // m
					 for (int n = threadIdx.y; n < 16; n += blockDim.y) { // n

					 for (int k = 0; k < 16; k++) {

					 su[m * 16 + k] = U_values[i][j][x + m][k];

					 }
					 }
					 }

					 __syncthreads();

					 for (int m = threadIdx.x; m < 16; m += blockDim.x) { // m
					 for (int n = threadIdx.y; n < 16; n += blockDim.y) { // n

					 for (int k = 0; k < 16; k++) {

					 int lidx = U_indices[i][j][x + m][k]; // blockDim.y threads are accessing this...
					 sw[k * 16 + n] = weights[lidx][y + n];

					 }
					 }
					 }
					 __syncthreads();*/

					for (int m = threadIdx.x; m < 16; m += blockDim.x) { // m
						for (int n = threadIdx.y; n < 16; n += blockDim.y) { // n

							float product = 0.0;

							for (int k = 0; k < 16; k++) {

								int lidx = U_indices[i][j][x + m][k];

								product += U_values[i][j][x + m][k]
										* weights[lidx][y + n];

								//product += su[m * 16 + k] * sw[k * 16 + n];
							}
							C[i][j][x + m][y + n] = product;
						}
					}

					/*wmma::fragment < wmma::matrix_a, 16, 16, 16, half, wmma::row_major > a_frag;
					 wmma::fragment < wmma::matrix_b, 16, 16, 16, half, wmma::row_major > b_frag;

					 wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

					 wmma::fill_fragment(c_frag, 0.0f);

					 wmma::load_matrix_sync(a_frag, &su[0][0], 16);
					 wmma::load_matrix_sync(b_frag, &sw[0][0], 16);

					 wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
					 wmma::store_matrix_sync(&so[0][0], c_frag, 16, wmma::mem_row_major);

					 for (int k = threadIdx.x; k < 16; k += blockDim.x) {
					 for (int l = threadIdx.y; l < 16; l += blockDim.y) {

					 C[i][j][x + k][y + l] = so[k][l];
					 }
					 } */

				}
			}
		}
	}
}

void Uw3_wmma_sparse_contraction_tensorcore_kernel_16x16_f32(
		torch::Tensor U_values, torch::Tensor U_indices, torch::Tensor U_nvals,
		torch::Tensor Uw_nvals, torch::Tensor W, torch::Tensor C) {

	const int nthreadsx = 3;
	const int nthreadsy = 32;

	dim3 blocks(U_values.size(0), U_values.size(1) / 4);

	dim3 grid(nthreadsx, nthreadsy);

	Uw3_wmma_sparse_contraction_tensorcore_kernel_16x16_f32<<<blocks, grid>>>(
			U_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			U_indices.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
			U_nvals.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			Uw_nvals.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			W.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

__global__ void Uw3_sparse_contraction_kernel_f32(
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> U_values,
		const torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> U_indices,
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> U_nvals,
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> Uw_indices,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> Uw_nvals,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
		torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {

	/***
	 *
	 * computes the matrix product U[i,j, :, :] \times w[:, :]
	 *
	 * U_values are the zero-padded non-sparse values of U with shape [48,48,48,16], with corresponding U_indices: [48,48,48,16]
	 *
	 * U_values actually stored as [48,48,16,48] as this allows for better contiguous memory access across threads.
	 *
	 * Exploits sparsity in U[i,j] as only 50% of these are non-zero. Also exploits sparsity in U[i,j,:, k] (6/48 sparsity).
	 *
	 *
	 * ***/

	//__shared__
	//float su[16][16]; // 48x16  -> 16x16
	//__shared__
	//float sw[16][16]; // 16*96 ->  16x16
	//extern __shared__ float buffer[];
	//float *su = &buffer[0];
	//float *sw = &buffer[16 * 16];
	//int nvals = Uw_nvals[i][j] -> can skip here as many U[i, j,:, :] are all zero, U[x, y, i, :] contains at most 6 non-zero vectors
	//50% of these blocks are doing matmuls on zero-valued matrices otherwise!
	for (int i = blockIdx.x; i < U_values.size(0); i += gridDim.x) {

		for (int j = blockIdx.y; j < U_values.size(1); j += gridDim.y) {

			if (Uw_nvals[i][j] == 0) {
				continue;
			}

			//sparsity over Uw_indices[i][j][x] is at most 6/48
			for (int x = threadIdx.x; x < Uw_nvals[i][j]; x += blockDim.x) {

				int Uw_idx = Uw_indices[i][j][x];

				int nvals = U_nvals[i][j][Uw_idx];

				for (int y = threadIdx.y; y < 96; y += blockDim.y) {

					float product = 0.0;

					for (int k = 0; k < nvals; k++) {

						int kidx = U_indices[i][j][Uw_idx][k];

						product += U_values[i][j][Uw_idx][k] * weights[kidx][y];
					}

					C[i][j][Uw_idx][y] = product;
				}
			}
		}
	}
}

void Uw3_sparse_contraction_kernel_16x16_f32(torch::Tensor U_values,
		torch::Tensor U_indices, torch::Tensor U_nvals,
		torch::Tensor Uw_indices, torch::Tensor Uw_nvals, torch::Tensor W,
		torch::Tensor C) {

	const int nthreadsx = 3;
	const int nthreadsy = 32;

	dim3 blocks(U_values.size(0), U_values.size(1) / 4); // 48,12

	dim3 grid(nthreadsx, nthreadsy);

	Uw3_sparse_contraction_kernel_f32<<<blocks, grid>>>(
			U_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			U_indices.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
			U_nvals.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			Uw_indices.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			Uw_nvals.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			W.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

__global__
void UwN3_sparse_contraction_kernel(
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Uw_dense,
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> Uw_indexes,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> Uw_nvals,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> features,
		torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {
	// Uw Shape: torch.Size([48, 48, 48, 96])
	// node features:           120, 48, 96

	//UwN = torch.einsum('...ic,bic ->bc...', Uw, node_feats)

//node features: natoms, nirreps, nfeatures

	__shared__
	float dots[96];

	for (int a = blockIdx.x; a < features.size(0); a += gridDim.x) { // natoms

		for (int i = blockIdx.y; i < 48; i += gridDim.y) { // nirreps

			/*note: due to sparsity pattern, later blocks over $j$ do 10 times less work than earlier blocks - need to sequentially space this*/
			for (int j = blockIdx.z; j < 48; j += gridDim.z) { // nirreps

				// nblocks[i] -> blockIdxs[i, j]
				// for j = blockIdx.z; j < nblocks[i]; j+= gridDim.z
				// int block_ij =  blockIdxs[i, j];
				// int kdim = Uw_nvals[i,j];
				// int idx = Uw_indexes[i][block_ij][k]

				int nval = Uw_nvals[i][j];

				if (nval == 0) {
					continue;
				}

				for (int l = threadIdx.x; l < 96; l += blockDim.x) {

					float dot = 0.0;

					for (int k = threadIdx.y; k < nval; k += blockDim.y) {
						int idx = Uw_indexes[i][j][k];

						dot += Uw_dense[i][j][idx][l] * features[a][idx][l];
					}

					atomicAdd(&C[a][i][j][l], dot);
				}
			}
		}
	}
}

void UwN3_sparse_contraction(torch::Tensor Uw_dense, torch::Tensor Uw_indexes,
		torch::Tensor Uw_nvals, torch::Tensor features, torch::Tensor C,
		const int nblocksx, const int nblocksy, const int nblocksz,
		const int nthreadsx, const int nthreadsy) {

	dim3 blocks(nblocksx, nblocksy, nblocksz);

	dim3 grid(nthreadsx, nthreadsy);

	UwN3_sparse_contraction_kernel<<<blocks, grid>>>(
			Uw_dense.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			Uw_indexes.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			Uw_nvals.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			features.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

/*
 for a in range(blockIdx.x, natoms, gridDim.x):
 for i in range(blockIdx.y, nirreps, gridDim.y):

 for j in range(0, n_irreps_per_block):
 irrep_j = irrepj_idxs[i][j]

 dot = 0.0
 for l in range (threadIdx.x, nfeatures, blockDim.x):

 for k in range (threadIdx.y, nirreps, blockDim.y):
 dot += c_tensor[a][i][irrep_j][l] * node_features[a][i][k]

 output[a][i][j][l] = dot

 */

/*
 __global__
 void UwN2_sparse_contraction_kernel(const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Uw,
 const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> irrep_idx_i,
 const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> irrep_idx_j,
 const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> features, torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> C) {
 // Uw Shape: torch.Size([120, 48, 48, 96])
 // node features:           120, 48, 96

 //UwN = torch.einsum('...ic,bic ->bc...', Uw, node_feats)

 //node features: natoms, nirreps, nfeatures

 tensor([[ 0,  1,  2,  ..., 45, 46, 47],
 [ 0,  1,  2,  ..., -1, -1, -1],
 [ 0,  1,  2,  ..., -1, -1, -1],
 ...,
 [ 0,  1,  2,  ..., -1, -1, -1],
 [ 0,  1,  2,  ..., -1, -1, -1],
 [ 0,  1,  4,  ..., -1, -1, -1]], device='cuda:0', dtype=torch.int32)

 for (int a = blockIdx.x; a < features.size(0); a += gridDim.x) { // natoms

 for (int i = blockIdx.y; i < irrep_idxs_i.size(0); i += gridDim.y) { // 1212 / nirreps_per_block

 for (int l = threadIdx.x; l < 96; l += blockDim.x) {

 float dot = 0.0;

 for (int j = threadIdx.y; j < irrep_idxs_i.size(1); j += blockDim.y) { // n_irreps_per_block

 int irrep_i_idx = irrep_idxs_i[i][j];
 int irrep_j_idx = irrep_idxs_j[i][j];

 if (irrep_i_idx == -1)
 continue;

 dot += Uw[a][irrep_i_idx][irrep_j_idx][k] * features[a][irrep_j_idx][k];

 }

 atomicAdd(&C[a][irrep_i_idx][l], dot);
 }

 }
 }
 }*/

__global__
void UwN2_dense_contraction_tc_kernel(
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> Uw,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> features,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> C) {

	__shared__ half
	su[16][16]; // 48x16  -> 16x16
	__shared__ half
	sw[16][16]; // 16*96 ->  16x16

	__shared__
	float so[16][16];  // 16x16

	__shared__
	float dots[48];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	// zero out shared memory
	for (int m = tidx; m < 16; m += blockDim.x) {

		for (int k = tidy; k < 16; k += blockDim.y) {
			sw[m][k] = __float2half(0.0f);
			su[m][k] = __float2half(0.0f);
		}
	}

	__syncthreads();

	for (int a = blockIdx.x; a < Uw.size(0); a += gridDim.x) { // natoms
		for (int b = blockIdx.y; b < Uw.size(1); b += gridDim.y) { // nfeatures

			wmma::fragment < wmma::matrix_a, 16, 16, 16, half, wmma::row_major
					> a_frag;
			wmma::fragment < wmma::matrix_b, 16, 16, 16, half, wmma::col_major
					> b_frag;
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

			if (threadIdx.x == 0) {
				for (int i = tidy; i < 48; i += blockDim.y) {
					dots[i] = 0.0;
				}
			}

			for (int x = 0; x < 48; x += 16) { // m

				for (int y = 0; y < 48; y += 16) { // k

					for (int m = tidx; m < 16; m += blockDim.x) {  // m
						for (int k = tidy; k < 16; k += blockDim.y) {  // k

							su[m][k] = __float2half(Uw[a][b][x + m][y + k]);
							sw[0][k] = __float2half(features[a][b][y + k]);
							so[m][k] = 0.0;
						}
					}

					__syncwarp();

					// Initialize the output to zero
					wmma::fill_fragment(c_frag, 0.0f);

					wmma::load_matrix_sync(a_frag, &su[0][0], 16);
					wmma::load_matrix_sync(b_frag, &sw[0][0], 16);

					// Perform the matrix multiplication
					wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

					wmma::store_matrix_sync(&so[0][0], c_frag, 16,
							wmma::mem_row_major);

					//copy so to global memory

					if (threadIdx.x == 0) {
						for (int i = tidy; i < 16; i += blockDim.y) {
							dots[x + i] += so[i][0];
						}
					}
					__syncwarp();
				}
			}

			if (threadIdx.x == 0) {

				for (int i = tidy; i < 48; i += blockDim.y) {
					C[a][b][i] = dots[i];
				}

			}
		}
	}
}

void UwN2_dense_contraction(torch::Tensor Uw3_dense, torch::Tensor features,
		torch::Tensor C, const int nblocksx, const int nblocksy,
		const int nthreadsx, const int nthreadsy) {

	dim3 blocks(nblocksx, nblocksy);

	dim3 grid(nthreadsx, nthreadsy);

	UwN2_dense_contraction_tc_kernel<<<blocks, grid, 256*8 + 48 *4>>>(
			Uw3_dense.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			features.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}
#define NWARPS 3

__global__ void multiwarp_test(
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> A,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> B,
		torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> C) {

	__shared__ half
	sA[NWARPS * 256]; // 48x16  -> 16x16
	__shared__ half
	sB[NWARPS * 256]; // 16*96 ->  16x16

	__shared__
	float sC[NWARPS * 256];  // 16x16

	int tidx = threadIdx.x; //nwarp identifier also
	int tidy = threadIdx.y;

	int warpid = tidx / 4;
	int laneid = tidx % NWARPS;

	wmma::fragment < wmma::matrix_a, 16, 16, 16, half, wmma::row_major
			> a_frag[NWARPS];
	wmma::fragment < wmma::matrix_b, 16, 16, 16, half, wmma::row_major
			> b_frag[NWARPS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[NWARPS];

	int start_idx = warpid * 256;

	for (int K = 0; K < A.size(1); K += 16) {

		for (int M = warpid * 16; M < A.size(0); M += NWARPS * 16) { // m -> NWARPS loop

			for (int m = laneid; m < 16; m += 4) { // m
				for (int k = threadIdx.y; k < 16; k += blockDim.y) { // k
					sA[start_idx + m * 16 + k] = __float2half(A[M + m][K + k]);
				}
			}

			for (int N = 0; N < B.size(1); N += 16) { // m -> NWARPS loop

				for (int k = laneid; k < 16; k += 4) {  // k
					for (int n = threadIdx.y; n < 16; n += blockDim.y) {  // n
						sB[start_idx + k * 16 + n] = __float2half(
								B[K + k][N + n]);
					}
				}

				__syncthreads();

				// Initialize the output to zero
				wmma::fill_fragment(c_frag[warpid], 0.0f);

				wmma::load_matrix_sync(a_frag[warpid], &sA[start_idx], 16);
				wmma::load_matrix_sync(b_frag[warpid], &sB[start_idx], 16);

				// Perform the matrix multiplication
				wmma::mma_sync(c_frag[warpid], a_frag[warpid], b_frag[warpid],
						c_frag[warpid]);

				wmma::store_matrix_sync(&sC[start_idx], c_frag[warpid], 16,
						wmma::mem_row_major);

				//copy sC to global memory

				__syncthreads();

				for (int m = laneid; m < 16; m += 4) {  // m
					for (int n = tidy; n < 16; n += blockDim.y) {  // n

						atomicAdd(&C[M + m][N + n], sC[start_idx + m * 16 + n]);
					}
				}
			}
		}
	}
}

void multiwarp_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

	dim3 blocks(1);

	dim3 grid(12, 8);

	multiwarp_test<<<blocks, grid, NWARPS * 256*2 + NWARPS * 256 *4>>>(
			A.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			B.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			C.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

