local classic = require 'classic'
local u = require 'dptycho.util'
local m = classic.module(...)

function m.fresnel(n,dx,dz,lambda)
  local qq2 = u.qq2(n,dx)
  local angle = qq2:mul(-lambda*math.pi*dz)
  local prop = torch.ZCudaTensor({n,n}):polar(1,angle:cuda())
  return prop
end

return m
