local ffi = require 'ffi'
local THNN = require 'nn.THNN'
local argcheck = require 'argcheck'
local THZNN = {}
local plot = require 'dptycho.io.plot'
local plt = plot()
-- load libTHZNN
THZNN.C = ffi.load(package.searchpath('libTHZNN', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function THZNN.getState()
   return THCState_ptr(cutorch.getState());
end

local THZNN_h = [[
TH_API void THNN_CudaTruncatedPoissonLikelihood_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask, THCudaTensor *output);
TH_API void THNN_CudaTruncatedPoissonLikelihood_GradientFactor(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask);
TH_API void THNN_CudaWSECriterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, float weight);
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
TH_API void THNN_ZCudaP_Mod_renorm(THCState *state, THZCudaTensor *self,
                                   THCudaTensor *fm, THCudaTensor *fdev,
                                   THCudaTensor *a, THCudaTensor *af,
                                   float renorm);
TH_API void THNN_ZCudaP_Mod_bg(THCState *state, THZCudaTensor *self,
                               THCudaTensor *fm, THCudaTensor *bg,
                               THCudaTensor *a, THCudaTensor *af, float renorm);
TH_API void THNN_ZCudaClipMinMax(THCState *state, THZCudaTensor *self_,
                                 THZCudaTensor *src1, float min, float max);

TH_API void THNN_ZCudaBatchedBilinearInterpolation(THCState *state,
                                                   THZCudaTensor *self_,
                                                   THZCudaTensor *src1,
                                                   float shiftx, float shifty);

TH_API void THNN_CudaBatchedBilinearInterpolation(THCState *state,
                                                  THCudaTensor *self_,
                                                  THCudaTensor *src1,
                                                  float shiftx, float shifty);
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

local shift = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   {name="shift", type = 'torch.FloatTensor'},
   call = function(dst, src, shift)
      -- pprint(shift)
      local shsq = shift:squeeze()
      -- pprint(shsq)
      local shiftx = shsq[1]
      local shifty = shsq[2]
      dst.THNN.BatchedBilinearInterpolation(dst:cdata(),src:cdata(),shiftx,shifty)
      return dst
   end
}

local shiftZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   {name="shift", type = 'torch.FloatTensor'},
   call = function(dst, src, shift)
      -- pprint(shift)
      local shsq = shift:squeeze()
      -- pprint(shsq)
      local shiftx = shsq[1]
      local shifty = shsq[2]
      dst.THNN.BatchedBilinearInterpolation(dst:cdata(),src:cdata(),shiftx,shifty)
      return dst
   end
}

local dxZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   {name="dxfw", type='torch.ZCudaTensor'},
   {name="dxbw", type='torch.ZCudaTensor'},
   call = function(dst, src, dxfw, dxbw)
      dxfw:shift(src,torch.FloatTensor({-1,0})):add(-1,src)
      dxbw:shift(src,torch.FloatTensor({1,0})):mul(-1):add(src)
      dst:add(dxfw,dxbw)
      dst[{{},{2,-2},{}}]:mul(0.5)
      return dst
   end
}

local dx_fwZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   call = function(dst, src)
      dst:shift(src,torch.FloatTensor({-1,0})):add(-1,src)
      return dst
   end
}

local dx_bwZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   call = function(dst, src)
      plt:plot(src[1]:zfloat(),'src')
      dst:shift(src,torch.FloatTensor({1,0}))
      plt:plot(dst[1]:zfloat(),'dst1')
      dst:mul(-1)
      plt:plot(dst[1]:zfloat(),'dst2')
      dst:add(src)
      plt:plot(dst[1]:zfloat(),'dst3')
      return dst
   end
}

local dy_fwZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   call = function(dst, src)
      dst:shift(src,torch.FloatTensor({0,-1})):add(-1,src)
      return dst
   end
}

local dy_bwZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   call = function(dst, src)
      dst:shift(src,torch.FloatTensor({0,1})):mul(-1):add(src)
      return dst
   end
}

local dx2Z = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   {name="dxfw", type='torch.ZCudaTensor'},
   {name="dxbw", type='torch.ZCudaTensor'},
   call = function(dst, src, dxfw, dxbw)
      dxfw:shift(src,torch.FloatTensor({-1,0}))
      dxbw:shift(src,torch.FloatTensor({1,0})):add(-2,src)
      dst:add(dxfw,dxbw)
      return dst
   end
}

local dxdyZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   {name="dxfw", type='torch.ZCudaTensor'},
   {name="dxbw", type='torch.ZCudaTensor'},
   {name="tmp1", type='torch.ZCudaTensor'},
   {name="tmp2", type='torch.ZCudaTensor'},
   call = function(dst, src, dxfw, dxbw, tmp1, tmp2)
      tmp1:dx(src,dxfw,dxbw)
      tmp2:dy(tmp1,dxfw,dxbw)
      dst:copy(tmp2)
      tmp1:dy(src,dxfw,dxbw)
      tmp2:dx(tmp1,dxfw,dxbw)
      dst:add(tmp2)
      dst:mul(0.5)
      return dst
   end
}

local dyZ = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   {name="dyfw", type='torch.ZCudaTensor'},
   {name="dybw", type='torch.ZCudaTensor'},
   call = function(dst, src, dyfw, dybw)
      dyfw:shift(src,torch.FloatTensor({0,-1})):add(-1,src)
      dybw:shift(src,torch.FloatTensor({0,1})):mul(-1):add(src)
      --
      dst:add(dyfw,dybw)
      dst[{{},{},{2,-2}}]:mul(0.5)
      return dst
   end
}

local dy2Z = argcheck{
   nonamed=true,
   {name="dst", type='torch.ZCudaTensor'},
   {name="src", type='torch.ZCudaTensor'},
   {name="dyfw", type='torch.ZCudaTensor'},
   {name="dybw", type='torch.ZCudaTensor'},
   call = function(dst, src, dyfw, dybw)
     dyfw:shift(src,torch.FloatTensor({0,-1}))
     dybw:shift(src,torch.FloatTensor({0,1})):add(-2,src)
     --
     dst:add(dyfw,dybw)
     return dst
   end
}

local dx = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   {name="dxfw", type='torch.CudaTensor'},
   {name="dxbw", type='torch.CudaTensor'},
   call = function(dst, src, dxfw, dxbw)
      dxfw:shift(src,torch.FloatTensor({-1,0}))
      dxbw:shift(src,torch.FloatTensor({1,0})):mul(-1)
      dst:add(dxfw,dxbw)
      dst[{{},{2,-2},{}}]:mul(0.5)
      return dst
   end
}

local dx_fw = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   call = function(dst, src)
      dst:shift(src,torch.FloatTensor({-1,0})):add(-1,src)
      return dst
   end
}

local dx_bw = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   call = function(dst, src)
      dst:shift(src,torch.FloatTensor({1,0})):mul(-1):add(src)
      return dst
   end
}

local dy_fw = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   call = function(dst, src)
      dst:shift(src,torch.FloatTensor({0,-1})):add(-1,src)
      return dst
   end
}

local dy_bw = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   call = function(dst, src)
      dst:shift(src,torch.FloatTensor({0,1})):mul(-1):add(src)
      return dst
   end
}

local dx2 = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   {name="dxfw", type='torch.CudaTensor'},
   {name="dxbw", type='torch.CudaTensor'},
   call = function(dst, src, dxfw, dxbw)
      dxfw:shift(src,torch.FloatTensor({-1,0}))
      dxbw:shift(src,torch.FloatTensor({1,0})):add(-2,src)
      dst:add(dxfw,dxbw)
      return dst
   end
}

local dxdy = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   {name="dxfw", type='torch.CudaTensor'},
   {name="dxbw", type='torch.CudaTensor'},
   {name="tmp1", type='torch.CudaTensor'},
   {name="tmp2", type='torch.CudaTensor'},
   call = function(dst, src, dxfw, dxbw, tmp1, tmp2)
      tmp1:dx(src,dxfw,dxbw)
      tmp2:dy(tmp1,dxfw,dxbw)
      dst:copy(tmp2)
      -- plt:plot(dst[1]:float(),'dxdy')
      tmp1:dy(src,dxfw,dxbw)
      tmp2:dx(tmp1,dxfw,dxbw)
      -- plt:plot(tmp2[1]:float(),'dydx')
      dst:add(tmp2)
      dst:mul(0.5)
      return dst
   end
}

local dy = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   {name="dyfw", type='torch.CudaTensor'},
   {name="dybw", type='torch.CudaTensor'},
   call = function(dst, src, dyfw, dybw)
      dyfw:shift(src,torch.FloatTensor({0,-1}))
      dybw:shift(src,torch.FloatTensor({0,1})):mul(-1)
      --
      dst:add(dyfw,dybw)
      dst[{{},{},{2,-2}}]:mul(0.5)
      return dst
   end
}

local dy2 = argcheck{
   nonamed=true,
   {name="dst", type='torch.CudaTensor'},
   {name="src", type='torch.CudaTensor'},
   {name="dyfw", type='torch.CudaTensor'},
   {name="dybw", type='torch.CudaTensor'},
   call = function(dst, src, dyfw, dybw)
     dyfw:shift(src,torch.FloatTensor({0,-1}))
     dybw:shift(src,torch.FloatTensor({0,1})):add(-2,src)
     --
     dst:add(dyfw,dybw)
     return dst
   end
}

rawset( meta, 'shift', shift)
rawset( meta, 'dx', dx)
rawset( meta, 'dy', dy)
rawset( meta, 'dx_fw', dx_fw)
rawset( meta, 'dy_fw', dy_fw)
rawset( meta, 'dx_bw', dx_bw)
rawset( meta, 'dy_bw', dy_bw)
rawset( meta, 'dx2', dx2)
rawset( meta, 'dy2', dy2)
rawset( meta, 'dxdy', dxdy)

rawset( Zmeta, 'shift', shiftZ)
rawset( Zmeta, 'dx', dxZ)
rawset( Zmeta, 'dy', dyZ)
rawset( Zmeta, 'dx_fw', dx_fwZ)
rawset( Zmeta, 'dy_fw', dy_fwZ)
rawset( Zmeta, 'dx_bw', dx_bwZ)
rawset( Zmeta, 'dy_bw', dy_bwZ)
rawset( Zmeta, 'dx2', dx2Z)
rawset( Zmeta, 'dy2', dy2Z)
rawset( Zmeta, 'dxdy', dxdyZ)
-- for name, f in pairs(Zfunctions) do
--   Zkernels[name] = f
--   Zmeta.THNN[name] = f
-- end

return THZNN
