local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THZNN = {}

-- load libTHZNN
THZNN.C = ffi.load(package.searchpath('libTHZNN', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function THZNN.getState()
   return THCState_ptr(cutorch.getState());
end

local THZNN_h = [[
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
]]

local preprocessed = string.gsub(THZNN_h, 'TH_API ', '')

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_Cuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end
local function extract_Zfunction_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_ZCuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end

ffi.cdef(preprocessed)

-- build function table
local function_names = extract_function_names(THZNN_h)
local Zfunction_names = extract_Zfunction_names(THZNN_h)

local kernels = THNN.kernels['torch.CudaTensor']
local meta = torch.getmetatable('torch.CudaTensor')

-- local Zkernels = THNN.kernels['torch.ZCudaTensor']
local Zmeta = torch.getmetatable('torch.ZCudaTensor')

local functions = THNN.bind(THZNN.C, function_names, 'Cuda', THZNN.getState)
local Zfunctions = THNN.bind(THZNN.C, Zfunction_names, 'ZCuda', THZNN.getState)
THNN.kernels['torch.ZCudaTensor'] = Zfunctions
Zmeta.THNN = Zfunctions

for name, f in pairs(functions) do
  kernels[name] = f
  meta.THNN[name] = f
end

-- for name, f in pairs(Zfunctions) do
--   Zkernels[name] = f
--   Zmeta.THNN[name] = f
-- end

return THZNN
