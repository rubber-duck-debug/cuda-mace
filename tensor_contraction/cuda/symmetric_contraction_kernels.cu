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
 
__global__ void uw3_contraction_kernel(torch::PackedTensorAccessor32<uint8_t, 3, torch::RestrictPtrTraits> UW3_indices, 
								torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> UW3_num_nonzeros,
								torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> UW3, 
								torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> UW2, 
								torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> UW1,
								torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> atom_types, 
								torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> X, 
								torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output3, 
								torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output2, 
								torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output1) {
	
	
	const int nchannels = X.size(2);
	const int natoms = X.size(0);
	const int nl = X.size(1);

	extern __shared__ char buffer[];

	size_t offset = 0;

	float *buffer_X = reinterpret_cast<float *>(buffer + offset);
	offset += blockDim.x * X.size(1) * sizeof(float);
	int * buffer_num_nonzeros = reinterpret_cast<int *>(buffer + offset);
	offset += UW3_num_nonzeros.size(0) * UW3_num_nonzeros.size(1) * sizeof(int);
	int * buffer_indices = reinterpret_cast<int *>(buffer + offset);
	offset += UW3_indices.size(0) * UW3_indices.size(1) * UW3_indices.size(2) * sizeof(int);


	if (threadIdx.z == 0) {
		for (int i = threadIdx.y; i < nl; i += blockDim.y) {
			for (int j = threadIdx.x; j < nl; j += blockDim.x ) {
				int n_nonzero = UW3_num_nonzeros[i][j];
				buffer_num_nonzeros[i * nl + j] = n_nonzero;

				for (int k = 0; k < n_nonzero; k ++) {
					buffer_indices[i * (nl * 3) + (k * nl) + j] = UW3_indices[i][j][k];
				}
			}
		}
	}

	for (int atom_id = blockIdx.x; atom_id < natoms; atom_id += gridDim.x) {

		int atom_type = atom_types[atom_id];

		for (int channel_id = threadIdx.x; channel_id < nchannels; channel_id += blockDim.x) {
			
			for (int i = threadIdx.y; i < nl; i += blockDim.y) {
				buffer_X[i * blockDim.x + threadIdx.x] = X[atom_id][i][channel_id];
			}

			__syncthreads();

			float sum1 = 0.0;

			for (int i = threadIdx.z; i < nl; i +=blockDim.z) {
				
				float Xi = buffer_X[i * nl + threadIdx.x];

				float uw2_i = UW1[i][atom_type][channel_id];
				
				float sum2 = 0.0;

				for (int j =  threadIdx.y; j < nl; j += blockDim.y) {

					//uint8_t n_nonzero = UW3_num_nonzeros[i][j];
					int n_nonzero = buffer_num_nonzeros[i * nl + j];
					
					//float uw2_ij = 0.0;
					float uw2_ij = UW2[i][j][atom_type][channel_id];

					float Xj = buffer_X[j * nl + threadIdx.x];

					float sum3 = 0.0; // uw2_ij;

					for (int k = 0; k < n_nonzero; k ++) {

						//int kdx = UW3_indices[i][j][k];
						int kdx = buffer_indices[i * (nl * 3) + (k * nl) + j];

						float Xk = buffer_X[kdx * nl + threadIdx.x];
						float uw3_k = UW3[i][j][kdx][atom_type][channel_id];

						sum3 += Xk * uw3_k;

					}
					sum2 += (sum3 + uw2_ij)  * Xj;

					output3[atom_id][i][j][channel_id] = sum3;
				}

				sum1 += (sum2 + uw2_i) * Xi;

				atomicAdd(&output2[atom_id][i][channel_id], sum2);

			}

			atomicAdd(&output1[atom_id][channel_id], sum1);
		}
	}
}


__global__ void add_uw2_kernel(torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> UW2, 
								torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> atom_types, 
								torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output3) {
	

	const int nchannels = output3.size(3);
	const int natoms = output3.size(0);
	const int nl = output3.size(1);


	for (int atom_id = blockIdx.x; atom_id < natoms; atom_id += gridDim.x) {

		uint8_t atom_type = atom_types[atom_id];

		for (int channel_id = threadIdx.x; channel_id < nchannels; channel_id += blockDim.x) {

			for (int i = threadIdx.z; i < nl; i +=blockDim.z) {
	
				for (int j =  threadIdx.y; j < nl; j += blockDim.y) {

					float uw2_ij = UW2[i][j][atom_type][channel_id];

					output3[atom_id][i][j][channel_id] += uw2_ij;

				}
			}
		}
	}
}

void add_uw2(
	const torch::Tensor UW2,
	const torch::Tensor atom_types, 
	torch::Tensor output3,
	const int nblockX = 1,
	const int nblockY = 1,
	const int nblockZ = 1,
	const int nthreadX = 32,
	const int nthreadY = 4,
	const int nthreadZ = 1) {

	int bx = nblockX;

	if (bx == 1)
	{
		bx = output3.size(0);
	}

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	add_uw2_kernel<<<blocks, grid>>>(
		UW2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
		output3.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

std::vector<torch::Tensor>  uw3_contraction(
	const torch::Tensor UW3_indices, // (nl + 1) ** 2, (nl + 1) ** 2, n_nonzero
	const torch::Tensor UW3_num_nonzeros, // (nl + 1) ** 2, (nl + 1) ** 2
	const torch::Tensor UW3, // (nl + 1) ** 2, (nl + 1) ** 2, (nl + 1) ** 2, nelements, nchannels
	const torch::Tensor UW2,
	const torch::Tensor UW1,
	const torch::Tensor atom_types, // natoms
	const torch::Tensor X, // natoms, (nl + 1) ** 2, nchannels
	const int nblockX = 1,
	const int nblockY = 1,
	const int nblockZ = 1,
	const int nthreadX = 32,
	const int nthreadY = 4,
	const int nthreadZ = 1)
{

	int bx = nblockX;

	if (bx == 1)
	{
		bx = X.size(0);
	}
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output3 = torch::empty({X.size(0), X.size(1), X.size(1), X.size(2)}, options);
	torch::Tensor output2 = torch::empty({X.size(0), X.size(1), X.size(2)}, options);
	torch::Tensor output1 = torch::empty({X.size(0), X.size(2)}, options);

	dim3 blocks(nblockX, nblockY, nblockZ);

	dim3 grid(nthreadX, nthreadY, nthreadZ);

	size_t shared_mem_amount = nthreadX * X.size(1) * sizeof(float); // X storage
	shared_mem_amount +=  UW3_num_nonzeros.numel() * sizeof(int);
	shared_mem_amount +=  UW3_indices.numel() * sizeof(int);

	uw3_contraction_kernel<<<blocks, grid, shared_mem_amount>>>(
		UW3_indices.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
		UW3_num_nonzeros.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
		UW3.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
		UW2.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		UW1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		atom_types.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
		X.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		output3.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
		output2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		output1.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

	return {output3, output2, output1};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("uw3_contraction", &uw3_contraction, "");
	m.def("add_uw2", &add_uw2, "");
	
}


