local Sqrt, parent = torch.class('znn.SqrtInPlace','nn.Sqrt')

function Sqrt:__init(b)
   parent.__init(self,b)
end

function Sqrt:updateOutput(input)
   self.eps = self.eps or 0
   input.THNN.Sqrt_updateOutput(
      input:cdata(),
      input:cdata(),
      self.eps
   )
   return self.output
end

function Sqrt:updateGradInput(input, gradOutput)
   input.THNN.Sqrt_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      input:sqrt():cdata()
   )
   return self.gradInput
end

return Sqrt
