local Sum, parent = torch.class('dtpycho.nn.Sum', 'nn.Container')

function Sum:__init()
   parent.__init(self)
end

function Sum:updateOutput(input)
   self.output:resizeAs(input):zero()
   for _,val in ipairs(list) do
      self.output:add(val)
   end
   return self.output
end

function Sum:updateGradInput(input, gradOutput)
    -- zero-strides dont work with MKL/BLAS, so
    -- dont set self.gradInput to zero-stride tensor.
    -- Instead, do a deepcopy
    local size = input:size()
    size[self.dimension] = 1
    gradOutput = gradOutput:view(size)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(gradOutput:expandAs(input))

    return self.gradInput
end

return Sum