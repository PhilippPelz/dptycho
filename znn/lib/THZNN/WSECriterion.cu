#include "THZNN.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/complex.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <stdlib.h>
struct mse_functor
{
  mse_functor() {}

  __host__ __device__ float operator()(const float &x, const float &y) const
  {
    float z = x-y;
    return z*z;
  }
};


void THNN_CudaWSECriterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, float weight)
{
  THAssert(THCudaTensor_checkGPU(state, 2, input, target));
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  float sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (float) 0,
    thrust::plus<float>(), mse_functor());

  // if (sizeAverage)
  //   sum /= size;
  // printf("sum: %f\n",sum);
  sum *= weight;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct mse_updateGradInput_functor
{
  const float norm;

  mse_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  __host__ __device__ float operator()(const float &x, const float &y) const
  {
    return norm * (x - y);
  }
};

void THNN_CudaWSECriterion_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *gradInput, float weight)
{
  THAssert(THCudaTensor_checkGPU(state, 3, input, target, gradInput));
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THCudaTensor_nElement(state, input);
  float norm = 2 * weight;

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    mse_updateGradInput_functor(norm));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
}

template <typename T> struct InvSigma : public thrust::unary_function<T, T> {
  T sigma;
  InvSigma(T _sigma) : sigma(_sigma) {}
  __host__ __device__ void operator()(T* out,T* in) { *out = T(1) / (*in + sigma); }
  __host__ __device__ void operator()(T* out) { *out = T(1) / (*out + sigma); }
};

TH_API void THNN_CudaInvSigma(THCState* state, THCudaTensor* self_, THCudaTensor* src, float sigma) {
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  THArgCheck(THCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, src), 3, "sizes do not match (self_,src1)");
  if (self_ == src) {
    if (!THCudaTensor_pointwiseApply1(state, self_, InvSigma<float>(sigma))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THCudaTensor_pointwiseApply2(state, self_, src, InvSigma<float>(sigma))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct ModProj {
	__device__ __forceinline__ void operator()(float* norm, float* abs, ccx* out) {
    if(*out != ccx(0)){
		    *out = (*out / *norm)* *abs ;
        // *out = thrust::polar(*abs,thrust::arg(*out));
    }
    else {
        *out = ccx(0);
    }
	}

	__device__ __forceinline__ void operator()(ccx* out, ccx* in1, float* in2) {
    if(*in1 != ccx(0)){
		    *out = *in1 / thrust::abs(*in1) * *in2;
    } else {
        *out = ccx(0);
    }
	}
};

void THNN_ZCudaP_Mod(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THCudaTensor *norm, THCudaTensor *f)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src1));
  THAssert(THCudaTensor_checkGPU(state, 1, f));
  THArgCheck(THZCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, f), 5, "sizes do not match (result,abs)");
  THArgCheck(THZCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, norm), 4, "sizes do not match (result,norm)");
  if (self_ == src1) {
    // self *= src2
    if (!THZCudaTensor_pointwiseApply3FFZ(state, norm, f, self_, ModProj())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    printf("out-of-place modulus projection not supported!");
    exit(1);
    // THZCudaTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    // if (!THZCudaTensor_pointwiseApply3ZZF(state, self_, src1, f, ModProj())) {
      // THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    // }
  }
}
