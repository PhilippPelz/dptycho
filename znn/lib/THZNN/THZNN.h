#include <THC/THC.h>
#include <THC/THZC.h>
#include <THC/THZCGeneral.cuh>
#include <THC/THCApply.cuh>
#include <THC/THZCApply.cuh>

TH_API void THNN_CudaWSECriterion_updateOutput(THCState *state,
                                               THCudaTensor *input,
                                               THCudaTensor *target,
                                               THCudaTensor *output,
                                               float weight);
TH_API void THNN_CudaWSECriterion_updateGradInput(THCState *state,
                                                  THCudaTensor *input,
                                                  THCudaTensor *target,
                                                  THCudaTensor *gradInput,
                                                  float weight);
TH_API void THNN_CudaInvSigma(THCState *state, THCudaTensor *self_,
                              THCudaTensor *src1, float sigma);
TH_API void THNN_ZCudaP_Mod(THCState *state, THZCudaTensor *self_,
                            THZCudaTensor *src1, THCudaTensor *norm,
                            THCudaTensor *f);
TH_API void THNN_ZCudaClipMinMax(THCState *state, THZCudaTensor *self_,
                                 THZCudaTensor *src1, float min, float max);

TH_API void THNN_ZCudaBatchedBilinearInterpolation(THCState *state,
                                                   THZCudaTensor *self_,
                                                   THZCudaTensor *src1,
                                                   float shiftx, float shifty);
