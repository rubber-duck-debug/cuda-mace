#include "spherical_harmonics.h"
#include "spherical_harmonics_wrapper.hpp"
#include "utils.h"

#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor SphericalHarmonicsAutograd::forward(
        AutogradContext *ctx,
        torch::Tensor xyz)
{  
    PUSH_RANGE("cuda_mace", 4)
    
    auto result = jit_spherical_harmonics(xyz);

    if (xyz.requires_grad())
    {
        ctx->save_for_backward({result[1]});
    }

    POP_RANGE

    return result[0];
}

variable_list SphericalHarmonicsAutograd::backward(AutogradContext *ctx, variable_list grad_outputs)
{
    PUSH_RANGE("cuda_mace", 4)

    auto saved_variables = ctx->get_saved_variables();

    torch::Tensor sph_deriv = saved_variables[0];
    
    torch::Tensor result = jit_spherical_harmonics_backward(sph_deriv, grad_outputs[0].contiguous());
    
    POP_RANGE

    return {result};
}

// wrapper class which we expose to the API.
torch::Tensor SphericalHarmonics::forward(
        torch::Tensor xyz)
{
    return SphericalHarmonicsAutograd::apply(xyz);
}

TORCH_LIBRARY(spherical_harmonics, m)
{
    m.class_<SphericalHarmonics>("SphericalHarmonics")
        .def(torch::init<>())

        .def("forward", &SphericalHarmonics::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<SphericalHarmonics> &self) -> std::vector<torch::Tensor>
            {
                return self->__getstate__();
            },
            [](const std::vector<torch::Tensor> &state) -> c10::intrusive_ptr<SphericalHarmonics>
            {
                auto obj = c10::make_intrusive<SphericalHarmonics>();
                obj->__setstate__(state);
                return obj;
            });
}