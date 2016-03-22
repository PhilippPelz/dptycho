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

]]

local preprocessed = string.gsub(THZNN_h, 'TH_API ', '')

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_Cuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end

ffi.cdef(preprocessed)

-- build function table
local function_names = extract_function_names(THZNN_h)

local kernels = THNN.kernels['torch.CudaTensor']
local meta = torch.getmetatable('torch.CudaTensor')

local functions = THNN.bind(THZNN.C, function_names, 'Cuda', THZNN.getState)

for name, f in pairs(functions) do
  kernels[name] = f
  meta.THNN[name] = f
end

return THZNN
