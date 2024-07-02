#include <torch/script.h>
#include <vector>

using namespace std;

std::vector<torch::Tensor> parse_elemental_embedding_v1(torch::Tensor elemental_embedding);
std::vector<torch::Tensor> parse_elemental_embedding_v2(torch::Tensor elemental_embedding);