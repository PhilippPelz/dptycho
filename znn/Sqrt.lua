local Sqrt, parent = torch.class('znn.Sqrt','nn.Sqrt')
local plot = require 'dptycho.io.plot'
local plt = plot()

function Sqrt:__init(b)
  parent.__init(self,b)
  self.gradInput = torch.CudaTensor()
  self.output = torch.CudaTensor()
end

function Sqrt:updateOutput(input)
  --  pprint(input)
   self.eps = self.eps or 0
   input.THNN.Sqrt_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.eps
   )
  --  print('in Sqrt:updateOutput')
  --  pprint(self.output)
  --  print('\n')
   return self.output
end

function Sqrt:updateGradInput(input, gradOutput)
  --  pprint(input)
  --  plt:plot(gradOutput:float(),'gradOutput')
  --  plt:plot(self.output:float(),'self.output')

  --  self.gradInput:resizeAs(input)
  --  print('in Sqrt:updateGradInput')
  --  pprint(self.gradInput)
  --  pprint(self.output)
  --  pprint(gradOutput)
   --
  --  print(gradOutput:max())
  --  print(gradOutput:min())
  --  print(self.output:max())
  --  print(self.output:min())

   input.THNN.Sqrt_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )

  --  print('\n')
  --  plt:plot(self.output:float(),'output')
  --  plt:plot(self.gradInput:float(),'gradOutput')
   return self.gradInput
end

return Sqrt
