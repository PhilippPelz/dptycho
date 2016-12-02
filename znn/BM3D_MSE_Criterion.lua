local nn = require 'nn'
local BM3D_MSE_Criterion, parent = torch.class('znn.BM3D_MSE_Criterion', 'nn.Criterion')
local bm3d = require 'bm3d'
require 'image'
require 'pprint'

function BM3D_MSE_Criterion:__init(tmp,grad,bm3d_params)
   parent.__init(self)
   self.update_filters_every = bm3d_params.denoise_interval
   self.O_sparse = tmp
   self.par = bm3d_params
   self.gradInput = grad
   self.amplitude = bm3d_params.amplitude
end

function BM3D_MSE_Criterion:updateOutput(input, it)
  if it % self.update_filters_every == 0 then
    local factor = 1
    local O = input[1][1]:zfloat()
    local rgb = u.complex2rgb(O)
    rgb:div(rgb:max())
    -- u.printf('rgb max: %g',rgb:max())
    local absmax = self.O[1][1]:abs():max()
    local O_basic = rgb:clone():zero()
    local O_denoised = rgb:clone():zero()
    image.save(string.format('bm3d_%03d.png',i),rgb)
    -- plt:plot(O,'noisy')
    bm3d.bm3d(self.par.sigma_denoise*factor,rgb,O_basic,O_denoised)
    -- u.printf('O_denoised max: %g',O_denoised:max())
    -- image.save(string.format('denoised%d.png',i),O_denoised:clone())
    local cx = u.rgb2complex(O_denoised)
    -- plt:plot(cx,'denoised')
    cx:mul(absmax)
    self.O_sparse[1][1]:copy(cx)
  end


  self.output_tensor = self.output_tensor or torch.FloatTensor(1)

  pprint(input)
  pprint(self.O_sparse)
  pprint(self.output_tensor)
  pprint(self.amplitude)
  
  input.THNN.WSECriterion_updateOutput(
    input:cdata(),
    self.O_sparse:cdata(),
    self.output_tensor:cdata(),
    self.amplitude
  )
  self.output = self.output_tensor[1]
  return self.output
end

function BM3D_MSE_Criterion:updateGradInput(input)
   input.THNN.WSECriterion_updateGradInput(
      input:cdata(),
      self.O_sparse:cdata(),
      self.gradInput:cdata(),
      self.amplitude
   )
   return self.gradInput
end

return BM3D_MSE_Criterion
