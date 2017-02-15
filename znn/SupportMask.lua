local plot = require 'dptycho.io.plot'
local u = require 'dptycho.util'
local plt = plot()

local c, parent = torch.class('znn.SupportMask', 'nn.Module')

function c:__init(size1,radius,type1)
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
  -- plt:plot(r:float(),'r')
  local mask = torch.lt(r, radius)
  -- plt:plot(mask:float(),'mask')
  mask = mask:repeatTensor(unpack(size)):float()
  -- pprint(mask)
  -- pprint(size1)
  self.weight = torch.ZFloatTensor(unpack(size1)):copy(mask):zcuda()
  self.output = torch.ZCudaTensor()
  -- pprint(self.weight)
end

function c:updateOutput(input)
  -- print('forward mask')
  -- pprint(input)
  -- pprint(self.weight)
  self.output:resizeAs(input):copy(input)
  return self.output:cmul(self.weight)
end

function c:updateGradInput(input, gradOutput)
  if self.update then
    gradOutput:cmul(self.weight)
  end
  self.gradInput = gradOutput
  return self.gradInput
end

return c
