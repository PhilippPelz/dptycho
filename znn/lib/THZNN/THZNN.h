#include <THC/THC.h>
#include <THC/THCApply.cuh>

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
