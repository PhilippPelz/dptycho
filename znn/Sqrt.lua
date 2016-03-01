local Sqrt, parent = torch.class('znn.Sqrt','nn.Sqrt')

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
   print('in Sqrt:updateOutput')
   pprint(self.output)
   print('\n')
   return self.output
end

function Sqrt:updateGradInput(input, gradOutput)
  --  pprint(input)
   input.THNN.Sqrt_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   print('in Sqrt:updateGradInput')
   pprint(self.gradInput)
   print('\n')
   return self.gradInput
end

return Sqrt
