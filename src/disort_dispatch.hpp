#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

#include <vector>

// disort
#include <cdisort213/cdisort.hpp>

namespace at::native {

using disort_fn = void (*)(at::TensorIterator& iter, int upward,
                           bool force_general, disort_state* ds,
                           disort_output* ds_out, at::Tensor* cuda_workspace);

DECLARE_DISPATCH(disort_fn, call_disort);

using special_boundary_fn = void (*)(torch::Tensor& output,
                                     const torch::Tensor& prop,
                                     const torch::Tensor& albedo,
                                     const disort_state& state,
                                     const std::vector<double>& angles,
                                     at::Tensor* cuda_workspace);

DECLARE_DISPATCH(special_boundary_fn, call_special_boundary);

}  // namespace at::native
