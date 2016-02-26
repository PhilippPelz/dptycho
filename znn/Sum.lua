require 'nn'
require 'dptycho.znn'
require 'pprint'
local Sum, parent = torch.class('znn.Sum', 'nn.Module')

function Sum:__init(dimension,ctor)
  --  parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.gradInput = ctor()
end

function Sum:updateOutput(input)
   return input:sum(self.dimension)
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
