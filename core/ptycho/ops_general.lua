local classic = require 'classic'
local u = require "dptycho.util"
local znn = require 'dptycho.znn'
local zt = require "ztorch.fcomplex"
local plot = require 'dptycho.io.plot'
local plt = plot()
local m = classic.class(...)

function m.static.P_Mod(x,abs,measured_abs)
  x.THNN.P_Mod(x:cdata(),x:cdata(),abs:cdata(),measured_abs:cdata())
end

function m.static.P_Mod_bg(x,fm,bg,a,af)
  x.THNN.P_Mod_bg(x:cdata(),fm:cdata(),bg:cdata(),a:cdata(),af:cdata(),1)
end

function m.static.P_Mod_renorm(x,fm,fdev,a,af,renorm)
  x.THNN.P_Mod_renorm(x:cdata(),fm:cdata(),fdev:cdata(),a:cdata(),af:cdata(),renorm)
end

function m.static.InvSigma(x,sigma)
  x.THNN.InvSigma(x:cdata(),x:cdata(),sigma)
end

function m.static.create_views(tensor,positions,view_size)
  local views = {}
  for i=1,positions:size(1) do
    local slice = {{},{},{positions[i][1],positions[i][1]+view_size-1},{positions[i][2],positions[i][2]+view_size-1}}
    -- pprint(slice)
    views[i] = tensor[slice]
  end
  return views
end

function m.static.P_F_without_background(z,a,a_exp,fm,fm_exp,zk_real_buffer, a_buffer1, a_buffer2, batch_param_table, iteration, power_threshold, fourier_mask,M,No,Np)
  -- print('P_F_without_background ' .. self.i)
  local abs = zk_real_buffer
  local da = a_buffer1
  local fdev = a_buffer2
  local err_fmag, renorm  = 0, 0
  local batch_start, batch_end, batch_size = table.unpack(batch_param_table)
  local module_error, mod_updates = 0, 0
  -- u.printf('z norm: %g',z:normall(2)^2)
  z:view_3D():fftBatched()
  -- u.printf('z norm: %g',z:normall(2)^2)
  for k=1,batch_size do
    local k_all = batch_start+k-1
    -- sum over probe and object modes -> 1x1xMxM
    local a_model = abs:normZ(z[k]):sum(1):sum(2)
    a_model:sqrt()
    -- pprint(abs)
    -- pprint(a[k_all])
    fdev[1][1]:add(a_model[1][1],-1,a[k_all])
    if k_all < 10 then
      -- plt:plot(fdev[1][1]:clone():fftshift():float(),'fdev')
      -- plt:plotcompare({a_model[1][1]:clone():fftshift():float():log(),a[k_all]:clone():fftshift():float():log()},{'a_model','a'})
    end
    da:pow(fdev,2)
    da:cmul(fm[k_all])
    err_fmag = da:sum()
    -- u.printf('err_fmag = %g',err_fmag)
    module_error = module_error + err_fmag

    local fdev_exp =  fdev[1][1]:view(1,1,M,M):expand(No,Np,M,M)
    local abs_exp =  a_model[1][1]:view(1,1,M,M):expand(No,Np,M,M)

    if err_fmag > power_threshold then
      renorm = math.sqrt(power_threshold/err_fmag)
      if fourier_mask ~= nil then
        fourier_mask:cmul(fm_exp[k_all])
      else
        fourier_mask = fm_exp[k_all]
      end
      m.P_Mod(z[k],abs_exp,a_exp[k_all])
      -- pprint(fourier_mask)
      -- m.P_Mod_renorm(z[k],fourier_mask,fdev_exp,a_exp[k_all],abs_exp,renorm)
      mod_updates = mod_updates + 1
    end
  end
  z:view_3D():ifftBatched()
  for i=1,6 do
    plt:plot(z[i][1][1]:clone()[{{80,400},{80,400}}],'exitwave '..i)
    -- plt:plotcompare({z[i][1][1]:clone():float():log(),a[k_all]:clone():fftshift():float():log()},{'a_model','a'})
  end
  -- u.printf('z norm: %g',z:normall(2)^2)
  return module_error, mod_updates
end

return m
