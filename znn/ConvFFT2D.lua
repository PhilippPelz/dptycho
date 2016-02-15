local c, parent = torch.class('dtpycho.nn.ConvFft2D', 'nn.Module')

function c:__init(filter,inv_filter)
   parent.__init(self)
   self.filter = filter
   self.inv_filter = inv_filter
end

function c:updateOutput(input)
   self.output:resizeAs(input):zero()
   self.output:fft(input):cmul(self.filter):ifft()
   return self.output
end

function c:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:fft(gradOutput):cmul(self.inv_filter):ifft()
    return self.gradInput
end

return c