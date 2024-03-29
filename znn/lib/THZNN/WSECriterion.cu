
#include "THZNN.h"
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/complex.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <stdlib.h>

template <typename T>
struct PoissonLikelihood : public thrust::unary_function<T, float> {
  __host__ __device__ float operator()(T x) {
    // thrust::get<0>(x) F[psi]
    // thrust::get<1>(x) a
    // thrust::get<2>(x) mask
    float intens = thrust::get<0>(x);
    float a = (float)thrust::get<1>(x);
    float m = (float)thrust::get<2>(x);
    return m*(intens - a * logf(intens));
  }
};

TH_API void THNN_CudaTruncatedPoissonLikelihood_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask, THCudaTensor *output)
{
  THAssert(THCudaTensor_checkGPU(state, 1, target));
  THAssert(THCudaTensor_checkGPU(state, 1, mask));
  THAssert(THCudaTensor_checkGPU(state, 1, output));
  THAssert(THCudaTensor_checkGPU(state, 1, input));
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);
  mask = THCudaTensor_newContiguous(state, mask);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> mask_data(THCudaTensor_data(state, mask));

  float sum = thrust::transform_reduce(
    #if CUDA_VERSION >= 7000
        thrust::cuda::par.on(THCState_getCurrentStream(state)),
    #endif
           thrust::make_zip_iterator(
               thrust::make_tuple(input_data, target_data, mask_data)),
           thrust::make_zip_iterator(
               thrust::make_tuple(input_data + size , target_data + size, mask_data + size)),
           PoissonLikelihood<thrust::tuple<float, float, float> >(),
           float(0), thrust::plus<float>());

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, mask);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct TruncatedPoissonLikelihood_GradientFactor_functor
{
  __host__ __device__ void operator()(float *intens, float *I_target, float *m) const
  {
    *intens = 2 * (*m) * (1- (*I_target) / (*intens + 1e-9));
  }
};

TH_API void THNN_CudaTruncatedPoissonLikelihood_GradientFactor(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask)
{
  THAssert(THCudaTensor_checkGPU(state, 1, input));
  THAssert(THCudaTensor_checkGPU(state, 2, target, mask));
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2, "sizes do not match (input,target)");
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, mask), 3, "sizes do not match (input,mask)");

  if (!THCudaTensor_pointwiseApply3(state, input, target, mask, TruncatedPoissonLikelihood_GradientFactor_functor())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
}

template <typename T>
struct EuclideanLoss : public thrust::unary_function<T, float> {
  __host__ __device__ float operator()(T x) {
    // thrust::get<0>(x) F[psi]
    // thrust::get<1>(x) a
    // thrust::get<2>(x) mask
    float a_model = thrust::get<0>(x);
    float a = (float)thrust::get<1>(x);
    float m = (float)thrust::get<2>(x);
    float da = (a_model - a );
    return m*da*da;
  }
};

TH_API void THNN_CudaEuclideanLoss_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask, THCudaTensor *output)
{
  THAssert(THCudaTensor_checkGPU(state, 1, target));
  THAssert(THCudaTensor_checkGPU(state, 1, mask));
  THAssert(THCudaTensor_checkGPU(state, 1, output));
  THAssert(THCudaTensor_checkGPU(state, 1, input));
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);
  mask = THCudaTensor_newContiguous(state, mask);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> mask_data(THCudaTensor_data(state, mask));

  float sum = thrust::transform_reduce(
    #if CUDA_VERSION >= 7000
        thrust::cuda::par.on(THCState_getCurrentStream(state)),
    #endif
           thrust::make_zip_iterator(
               thrust::make_tuple(input_data, target_data, mask_data)),
           thrust::make_zip_iterator(
               thrust::make_tuple(input_data + size , target_data + size, mask_data + size)),
           EuclideanLoss<thrust::tuple<float, float, float> >(),
           float(0), thrust::plus<float>());

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, mask);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct EuclideanLoss_GradientFactor_functor
{
  __host__ __device__ void operator()(float *a_model, float *a, float *m) const
  {
    if(*a < 1e-6 && *a_model < 1e-6){
      *a = (*m);
    } else {
      *a = (*m) * (1- (*a_model) / (*a + 1e-9));
    }
  }
};

TH_API void THNN_CudaEuclideanLoss_GradientFactor(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask)
{
  THAssert(THCudaTensor_checkGPU(state, 1, input));
  THAssert(THCudaTensor_checkGPU(state, 2, target, mask));
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2, "sizes do not match (input,target)");
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, mask), 3, "sizes do not match (input,mask)");

  if (!THCudaTensor_pointwiseApply3(state, input, target, mask, EuclideanLoss_GradientFactor_functor())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
}

struct zmse_functor
{
  __host__ __device__ float operator()(const ccx& x, const ccx& y) const
  {
    float a = thrust::abs(x - y);
    return a*a;
  }
};

TH_API void THNN_ZCudaWSECriterion_updateOutput(THCState *state, THZCudaTensor *input, THZCudaTensor *target, THFloatTensor *output, float weight)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, input, target));
  THArgCheck(THZCudaTensor_nElement(state, input) == THZCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THZCudaTensor_nElement(state, input);

  input = THZCudaTensor_newContiguous(state, input);
  target = THZCudaTensor_newContiguous(state, target);

  thrust::device_ptr<ccx> input_data((ccx*)THZCudaTensor_data(state, input));
  thrust::device_ptr<ccx> target_data((ccx*)THZCudaTensor_data(state, target));
  float sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (float) 0,
    thrust::plus<float>(), zmse_functor());

  // if (sizeAverage)
  //   sum /= size;
  // printf("sum: %f\n",sum);
  sum *= weight;

  THZCudaTensor_free(state, input);
  THZCudaTensor_free(state, target);

  THFloatTensor_set1d(output, 0, sum);
}

struct zmse_updateGradInput_functor
{
  const float norm;

  zmse_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  __host__ __device__ ccx operator()(const ccx& x, const ccx& y) const
  {
    return norm * (x - y);
  }
};

void THNN_ZCudaWSECriterion_updateGradInput(THCState *state, THZCudaTensor *input, THZCudaTensor *target, THZCudaTensor *gradInput, float weight)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, input, target, gradInput));
  THArgCheck(THZCudaTensor_nElement(state, input) == THZCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THZCudaTensor_nElement(state, input);
  float norm = 2 * weight;

  input = THZCudaTensor_newContiguous(state, input);
  target = THZCudaTensor_newContiguous(state, target);

  THZCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<ccx> input_data((ccx*)THZCudaTensor_data(state, input));
  thrust::device_ptr<ccx> target_data((ccx*)THZCudaTensor_data(state, target));
  thrust::device_ptr<ccx> gradInput_data((ccx*)THZCudaTensor_data(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    zmse_updateGradInput_functor(norm));

  THZCudaTensor_free(state, input);
  THZCudaTensor_free(state, target);
}

struct mse_functor
{
  __host__ __device__ float operator()(const float &x, const float &y) const
  {
    return (x - y)*(x - y);
  }
};

TH_API void THNN_CudaWSECriterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, float weight)
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

struct ModProj_renorm_bg {
  float renorm;
  ModProj_renorm_bg(float _renorm) : renorm(_renorm) {}
	__device__ __forceinline__ void operator()(float* fm, float* bg, float* a, float* af, ccx* out) {
      //fm = (1-fmask) + fmask*(fmag + fdev*renorm)/(af + 1e-10)
      // float fac = (1-*fm) + *fm * (*a+*fdev* (renorm)) / (*af + 1e-6f);
      float a0 = fmaxf(0,*a * *a - *bg);
      float fac = (1-*fm) + *fm * sqrtf(a0 / (*af * *af + 1e-6f));
	    *out = *out * fac ;
	}
};

void THNN_ZCudaP_Mod_bg(THCState *state, THZCudaTensor *self, THCudaTensor *fm, THCudaTensor * bg, THCudaTensor * a, THCudaTensor * af, float renorm )
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self));
  THAssert(THCudaTensor_checkGPU(state, 3, fm,bg,a,af));
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, fm), 3, "sizes do not match (result,fm)");
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, bg), 4, "sizes do not match (result,fdev)");
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, a), 5, "sizes do not match (result,a)");
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, af), 6, "sizes do not match (result,af)");

  if (!THZCudaTensor_pointwiseApply5FFFFZ(state, fm, bg, a, af, self, ModProj_renorm_bg(renorm))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
}

// --   fmag = np.sqrt(np.abs(I))
// -- af=np.sqrt(af2)
// -- fdev = af - fmag
// -- err_fmag = np.sum(fmask*fdev**2)/fmask.sum()
//
// -- renorm = np.sqrt(pbound / err_fmag) if pbound is not None else 0.0 # don't know if that is correct
// -- fm = (1-fmask) + fmask*(fmag + fdev*renorm)/(af + 1e-10)
struct ModProj_renorm {
  float renorm;
  ModProj_renorm(float _renorm) : renorm(_renorm) {}
	__device__ __forceinline__ void operator()(float* fm, float* fdev, float* a, float* a_model, ccx* out) {
      //fm = (1-fmask) + fmask*(fmag + fdev*renorm)/(a_model + 1e-10)
      float fac = (1-*fm) + *fm * (*a+*fdev* (renorm)) / (*a_model + 1e-6f);
      // float fac = (1.0-*fm) + *fm * (*a) / (*a_model + 1e-5f);
	    *out = *out * fac ;
	}
};

void THNN_ZCudaP_Mod_renorm(THCState *state, THZCudaTensor *self, THCudaTensor *fm, THCudaTensor * fdev, THCudaTensor * a, THCudaTensor * a_model, float renorm )
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self));
  THAssert(THCudaTensor_checkGPU(state, 3, fm,fdev,a,a_model));
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, fm), 3, "sizes do not match (result,fm)");
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, fdev), 4, "sizes do not match (result,fdev)");
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, a), 5, "sizes do not match (result,a)");
  THArgCheck(THZCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, a_model), 6, "sizes do not match (result,a_model)");

  if (!THZCudaTensor_pointwiseApply5FFFFZ(state, fm, fdev, a, a_model, self, ModProj_renorm(renorm))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
}

struct ModProj {
  ModProj() {}
	__device__ __forceinline__ void operator()(float* fm, float* a, float* a_model, float* a_model1, ccx* out) {
      float fac = (1.0-*fm) + *fm * (*a) / (*a_model + 1e-5f);
	    *out = *out * fac ;
	}
};

void THNN_ZCudaP_Mod(THCState *state, THZCudaTensor *self_, THCudaTensor *fm, THCudaTensor *a, THCudaTensor *a_model)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_));
  THAssert(THCudaTensor_checkGPU(state, 1, a));
  THArgCheck(THZCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, a), 5, "sizes do not match (result,abs)");
  THArgCheck(THZCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, a_model), 4, "sizes do not match (result,norm)");
  if (!THZCudaTensor_pointwiseApply5FFFFZ(state, fm, a, a_model,a_model, self_, ModProj())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
}

struct TensorClipMax {
  float _max;
  float _min;
  TensorClipMax(float min,float max) : _max(max), _min(min) {}
  __device__ __forceinline__ void operator()(ccx* o, ccx* i) const {
    float ro = i->real();
    float io = i->imag();
    if(ro>_max) ro = _max;
    if(io>_max) io = _max;
    if(ro<_min) ro = _min;
    if(io<_min) io = _min;
    *o = ccx(ro,io);
  }

  __device__ __forceinline__ void operator()(ccx* v) const {
    float ro = v->real();
    float io = v->imag();
    if(ro>_max) ro = _max;
    if(io>_max) io = _max;
    if(ro<_min) ro = _min;
    if(io<_min) io = _min;
    *v = ccx(ro,io);
  }
};

void THNN_ZCudaClipMinMax(THCState *state, THZCudaTensor *self_,
                              THZCudaTensor *src, float min, float max){
THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
if (self_ == src) {
  if (!THZCudaTensor_pointwiseApply1(state, self_, TensorClipMax(min,max))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
} else {
  THZCudaTensor_resizeAs(state, self_, src);

  if (!THZCudaTensor_pointwiseApply2(state, self_, src, TensorClipMax(min,max))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
}

THZCudaCheck(cudaGetLastError());
}

__device__ int ind3d(int x, int y, int z, int X, int Y){
  return z * (X * Y) + y * X + x;
}

__global__ void batched_bilinear_interpolation_kernelZ(ccx * dest, ccx * src, const int X, const int Y, int shx, int shy, float shiftx, float shifty)
{
   const int z = blockIdx.z;
   const int x = threadIdx.x + blockDim.x * blockIdx.x;
   const int y = threadIdx.y + blockDim.y * blockIdx.y;

   if ((x<X)&&(y<Y)) {
    const int    ind_x = x+shx;
    const float  a     = shiftx;

    const int    ind_y = y+shy;
    const float  b     = shifty;

    ccx h00, h01, h10, h11;
    if (((ind_x)   < X)&&((ind_y)   < Y) && ((ind_x)   >= 0)&&((ind_y)   >= 0)) h00 = src[ind3d(ind_x,ind_y,z,X,Y)];  else    h00 =0;
    if (((ind_x+1) < X)&&((ind_y)   < Y) && ((ind_x+1) >= 0)&&((ind_y)   > 0)) h10 = src[ind3d(ind_x+1,ind_y,z,X,Y)];     else    h10 = 0;
    if (((ind_x)   < X)&&((ind_y+1) < Y) && ((ind_x)   >= 0)&&((ind_y+1) >= 0)) h01 = src[ind3d(ind_x,ind_y+1,z,X,Y)];   else    h01 = 0;
    if (((ind_x+1) < X)&&((ind_y+1) < Y) && ((ind_x+1) >= 0)&&((ind_y+1) >= 0)) h11 = src[ind3d(ind_x+1,ind_y+1,z,X,Y)]; else    h11 = 0;

    dest[ind3d(x,y,z,X,Y)] = (1-a)*(1-b)*h00 +
                              (a)*(1-b)*h10 +
                              (1-a)*(b)*h01 +
                              a*b*h11;
   }
}

TH_API void THNN_ZCudaBatchedBilinearInterpolation(THCState *state,
                                                   THZCudaTensor *self_,
                                                   THZCudaTensor *src1,
                                                   float u, float v){
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src1));
  THArgCheck(self_ != src1, 2, "source and destination must be two distinct tensors");
  THArgCheck(THZCudaTensor_nElement(state, self_) == THZCudaTensor_nElement(state, src1), 3, "sizes do not match (self_,src1)");
  u *= -1;
  v *=-1;
  int xi = 0, yi = 0;
  while(u < 0.0f){
    u+=1.0f;
    xi--;
  }
  while(v < 0.0f){
    v+=1.0f;
    yi--;
  }
  while(u > 1.0f){
    u-=1.0f;
    xi++;
  }
  while(v > 1.0f){
    v-=1.0f;
    yi++;
  }
  long iz = THZCudaTensor_size(state, self_, 0);
  long iy = THZCudaTensor_size(state, self_, 1);
  long ix = THZCudaTensor_size(state, self_, 2);
  // ZTensorInfo<unsigned int> aInfo(state, self_);
  // ZTensorInfo<unsigned int> bInfo(state, a);
  int dest_cont = THZCudaTensor_isContiguous(state,self_);
  int src_cont = THZCudaTensor_isContiguous(state,src1);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((ix / threadsPerBlock.x) + 1, (iy / threadsPerBlock.y) + 1, iz);

  // printf("ix,iy,iz,xi,u,yi,v = %-5d   %-5d  %-5d  %-5d  %-5f  %-5d  %-5f  dest_cont: %d  src_cont: %d blocks (%d,%d,%d) max_index (%d,%d)\n",ix,iy,iz,xi,u,yi,v,dest_cont,src_cont,(ix / threadsPerBlock.x) + 1,(iy / threadsPerBlock.y) + 1,iz,((ix / threadsPerBlock.x) + 1)*16,( (iy / threadsPerBlock.y) + 1)*16);

  ccx* dest_data = (ccx*)THZCudaTensor_data(state, self_);
  ccx* src_data = (ccx*)THZCudaTensor_data(state, src1);

  batched_bilinear_interpolation_kernelZ<<<numBlocks,threadsPerBlock,0,THCState_getCurrentStream(state)>>>(dest_data,src_data,ix,iy,xi,yi,u,v);
}

__global__ void batched_bilinear_interpolation_kernel(float * dest, float * src, const int X, const int Y, int shx, int shy, float shiftx, float shifty)
{
  const int z = blockIdx.z;
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x<X)&&(y<Y)) {
   const int    ind_x = x+shx;
   const float  a     = shiftx;

   const int    ind_y = y+shy;
   const float  b     = shifty;

   float h00, h01, h10, h11;
   if (((ind_x)   < X)&&((ind_y)   < Y) && ((ind_x)   >= 0)&&((ind_y)   >= 0)) h00 = src[ind3d(ind_x,ind_y,z,X,Y)];  else    h00 =0;
   if (((ind_x+1) < X)&&((ind_y)   < Y) && ((ind_x+1) >= 0)&&((ind_y)   > 0)) h10 = src[ind3d(ind_x+1,ind_y,z,X,Y)];     else    h10 = 0;
   if (((ind_x)   < X)&&((ind_y+1) < Y) && ((ind_x)   >= 0)&&((ind_y+1) >= 0)) h01 = src[ind3d(ind_x,ind_y+1,z,X,Y)];   else    h01 = 0;
   if (((ind_x+1) < X)&&((ind_y+1) < Y) && ((ind_x+1) >= 0)&&((ind_y+1) >= 0)) h11 = src[ind3d(ind_x+1,ind_y+1,z,X,Y)]; else    h11 = 0;

   dest[ind3d(x,y,z,X,Y)] = (1-a)*(1-b)*h00 +
                             (a)*(1-b)*h10 +
                             (1-a)*(b)*h01 +
                             a*b*h11;
  }
}

TH_API void THNN_CudaBatchedBilinearInterpolation(THCState *state,
                                                   THCudaTensor *self_,
                                                   THCudaTensor *src1,
                                                   float u, float v){
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src1));
  THArgCheck(self_ != src1, 2, "source and destination must be two distinct tensors");
  THArgCheck(THCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, src1), 3, "sizes do not match (self_,src1)");
  u *= -1;
  v *=-1;
  int xi = 0, yi = 0;
  while(u < 0.0f){
    u+=1.0f;
    xi--;
  }
  while(v < 0.0f){
    v+=1.0f;
    yi--;
  }
  while(u > 1.0f){
    u-=1.0f;
    xi++;
  }
  while(v > 1.0f){
    v-=1.0f;
    yi++;
  }
  long iz = THCudaTensor_size(state, self_, 0);
  long iy = THCudaTensor_size(state, self_, 1);
  long ix = THCudaTensor_size(state, self_, 2);

  // printf("xi,u,yi,v = %d,%f,%d,%f\n",xi,u,yi,v);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((ix / threadsPerBlock.x)+ 1, (iy / threadsPerBlock.y)+ 1, iz);

  float* dest_data = THCudaTensor_data(state, self_);
  float* src_data = THCudaTensor_data(state, src1);

  batched_bilinear_interpolation_kernel<<<numBlocks,threadsPerBlock,0,THCState_getCurrentStream(state)>>>(dest_data,src_data,ix,iy,xi,yi,u,v);
}












































int i = 0;
