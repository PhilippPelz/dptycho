local plot = require 'dptycho.io.plot'
local u = require 'dptycho.util'
local plt = plot()

local c, parent = torch.class('znn.SupportMask', 'nn.Module')

function c:__init(size1,radius,type1,fft_shifted,outside_scaling)
  local fft_shift = fft_shifted or false
  local out_scaling = outside_scaling or 1e-2
  local type = type1 or 'torch.ZCudaTensor'
  parent.__init(self)
  local size = u.copytable(size1)
  local Nx, Ny = size[#size], size[#size-1]
  -- print(Nx,Ny)
  -- pprint(size)
  size[#size] = 1
  size[#size-1] = 1
  -- float2 --> complex
  -- size[#size+1] = 2
  -- print(Nx,Ny)
  local x = torch.repeatTensor(torch.linspace(-Nx/2,Nx/2,Nx),Ny,1)
  -- pprint(x)
  local y = x:clone():t()
  local r = (x:pow(2) + y:pow(2)):sqrt()

  local r_out = r:clone():add(-radius):abs():float()
  -- plt:plot(r_out,'r_out')
  r_out:div(-2*(out_scaling/2.35482)^2):exp()
  -- plt:plot(r_out,'r_out')

  local inside = torch.le(r, radius):cuda()
  local outside1 = torch.gt(r, radius):float():cmul(r_out)
  local outside = outside1:cuda()

  self.mask = inside:add(outside)

  if fft_shift then
    self.mask:fftshift()
  end
  -- plt:plot(self.mask:float(),'mask')
  -- mask = mask:repeatTensor(unpack(size)):float()
  -- pprint(mask)
  -- pprint(size1)
  -- self.weight = torch.ZFloatTensor(unpack(size1)):copy(mask):zcuda()
  self.output = torch.ZCudaTensor()
  -- pprint(self.weight)
end

function c:updateOutput(input)
  local size = input:size():totable()
  for i=1,#size-2 do
    size[i] = 1
  end
  -- print('SupportMask updateOutput',size)
  self.weight = self.mask:view(table.unpack(size)):expandAs(input)
  -- print('forward mask')
  -- pprint(input)
  -- pprint(self.weight)
  self.output:resizeAs(input):copy(input)
  self.output:cmul(self.weight)
  return input:cmul(self.weight)
end

function c:updateGradInput(input, gradOutput)
  if self.update then
    gradOutput:cmul(self.weight)
  end
  self.gradInput = gradOutput
  return self.gradInput
end

return c
