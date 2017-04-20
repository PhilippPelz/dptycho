local classic = require 'classic'
local u = require "dptycho.util"
local znn = require 'dptycho.znn'
local zt = require "ztorch.fcomplex"
local plot = require 'dptycho.io.plot'
local plt = plot()
require 'hdf5'
local m = classic.class(...)

function m.static.P_Mod(x,fourier_mask,abs,measured_abs)
  x.THNN.P_Mod(x:cdata(),fourier_mask:cdata(),x:cdata(),abs:cdata(),measured_abs:cdata())
end

function m.static.P_Mod_bg(x,fm,bg,a,af)
  x.THNN.P_Mod_bg(x:cdata(),fm:cdata(),bg:cdata(),a:cdata(),af:cdata(),1)
end

function m.static.P_Mod_renorm(x,fm,fdev,a,a_model_exp,renorm)
  x.THNN.P_Mod_renorm(x:cdata(),fm:cdata(),fdev:cdata(),a:cdata(),a_model_exp:cdata(),renorm)
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
i = 1

function m.static.filter_object(O,object_highpass)
  -- pprint(self.O_tmp_PQstore)
  local O_fluence = O:normall(2)
  -- print('filtering object')

  O:view_3D():fftBatched()

  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft unfiltered ',self.save_path .. self.i..' P filtered ',false)
  O:cmul(object_highpass)
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft filtered ',self.save_path .. self.i..' P filtered ',false)
  O:view_3D():ifftBatched()

  local O_fluence_new = O:normall(2)
  u.printf('O_fluence/O_fluence_new = %g',O_fluence/O_fluence_new)
  O:mul(O_fluence/O_fluence_new)
end

function m.static.filter_probe(P,probe_lowpass)
  local P_fluence = P:normall(2)
  -- print('filtering probe')
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P',self.save_path .. self.i..' P',false)
  P:view_3D():fftBatched()
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft unfiltered ',self.save_path .. self.i..' P filtered ',false)
  P:cmul(probe_lowpass)
  -- plt:plot(self.P[1][1]:zfloat():abs():log(),self.i..' P fft filtered ',self.save_path .. self.i..' P filtered ',false)
  P:view_3D():ifftBatched()
  local P_fluence_new = P:normall(2)
  -- u.printf('P_fluence/P_fluence_new = %g',P_fluence/P_fluence_new)
  P:mul(P_fluence/P_fluence_new)
  -- plt:plot(self.P[1][1]:zfloat(),self.i..' P filtered ',self.save_path .. self.i..' P filtered ',false)
end

function m.static.P_F_without_background(z,a,a_exp,fm,fm_exp,zk_real_buffer, a_buffer1, a_buffer2, batch_param_table, iteration, power_threshold, fourier_mask,M,No,Np)
  -- print('P_F_without_background ' .. self.i)
  local abs = zk_real_buffer
  local da = a_buffer1
  local fdev = a_buffer2
  local err_fmag, renorm  = 0, 0
  local batch_start, batch_end, batch_size = table.unpack(batch_param_table)
  local module_error, mod_updates = 0, 0
  if iteration % 15 == 0 then
    save = a:clone():float()
  end
  -- u.printf('z norm: %g',z:normall(2)^2)
  -- for i=80,120 do
    -- plt:plot(z[i][1][1],'before exitwave '..i)
    -- plt:plotcompare({z[i][1][1]:clone():float():log(),a[i]:clone():fftshift():float():log()},{'a_model','a'})
  -- end
  -- plt:plot(z[36][1][1],'before exitwave '..36)
  z:view_3D():fftBatched()
  -- plt:plot(z[36][1][1],'after  exitwave '..36)
  -- u.printf('z norm: %g',z:normall(2)^2)
  for k=1,batch_size do
    local k_all = batch_start+k-1
    -- sum over probe and object modes -> 1x1xMxM
    local a_model = abs:normZ(z[k]):sum(1):sum(2)
    a_model:sqrt()
    -- pprint(abs)
    -- pprint(a[k_all])
    fdev[1][1]:add(a_model[1][1],-1,a[k_all])
    if iteration % 15 == 0 then
    -- self,imgs, title, suptitle, savepath
      save[k_all]:copy(fdev[1][1]:float())
      -- plt:plot(fdev[1][1]:clone():cdiv(a[k_all]):fftshift():float(),'fdev relative',path .. '/fdev/'.. string.format('%d_it%d_fdev',k_all,i),false)
      -- plt:plotcompare({a_model[1][1]:clone():fftshift():float(),a[k_all]:clone():fftshift():float()},{'a_model '..k_all,'a '..k_all},'',path .. '/a/'.. string.format('%d_it%d_a',k_all,i),false)
    end
    da:pow(fdev,2)
    da:cmul(fm[k_all])
    err_fmag = da:sum()
    -- u.printf('err_fmag = %g',err_fmag)
    module_error = module_error + err_fmag

    local fdev_exp =  fdev[1][1]:view(1,1,M,M):expand(No,Np,M,M)
    local a_model_exp =  a_model[1][1]:view(1,1,M,M):expand(No,Np,M,M)

    if err_fmag > power_threshold then
      renorm = math.sqrt(power_threshold/err_fmag)
      if fourier_mask then
        fourier_mask:cmul(fm_exp[k_all])
      else
        fourier_mask = fm_exp[k_all]
      end
      -- m.P_Mod(z[k],a_model_exp,a_exp[k_all])
      -- pprint(fourier_mask)
      -- plt:plot(z[k][1][1],'before P_Mod_renorm '..k)
      m.P_Mod_renorm(z[k],fourier_mask,fdev_exp,a_exp[k_all],a_model_exp,renorm)
      -- plt:plot(z[k][1][1],'after  P_Mod_renorm '..k)
      mod_updates = mod_updates + 1
    end
  end
  z:view_3D():ifftBatched()
  for i=11,13 do
    -- plt:plot(z[i][1][1],'after exitwave '..i)
    -- plt:plotcompare({z[i][1][1]:clone():float():log(),a[i]:clone():fftshift():float():log()},{'a_model','a'})
  end
  -- u.printf('z norm: %g',z:normall(2)^2)
  i = i + 1
  if iteration % 15 == 0 then
    local f = hdf5.open(path.. 'fdev1.h5')
    local options = hdf5.DataSetOptions()
    options:setChunked(1,128,128)
    options:setDeflate(8)
    f:write('/fdev',save,options)
    f:close()
  end
  return module_error, mod_updates
end

function m.static.P_F_without_background_basic(z,a_exp,fm_exp,zk_real_buffer,  fourier_mask,M,No,Np)
  -- print('P_F_without_background ' .. self.i)
  local abs = zk_real_buffer
  local err_fmag, renorm  = 0, 0
  local module_error, mod_updates = 0, 0
  z:view_3D():fftBatched()
  local a_model = abs:normZ(z):sum(1):sum(2)
  a_model:sqrt()
  local a_model_exp =  a_model[1][1]:view(1,1,M,M):expand(No,Np,M,M)

  if fourier_mask then
    fourier_mask:cmul(fm_exp)
  else
    fourier_mask = fm_exp
  end

  plt:plot(z[1][1],'before P_Mod ')
  m.P_Mod(z,fourier_mask,a_exp,a_model_exp)
  plt:plot(z[1][1],'after P_Mod ')

  z:view_3D():ifftBatched()
  return module_error, mod_updates
end

-- function m.static.P_F_without_background(z,a,a_exp,fm,fm_exp,zk_real_buffer, a_buffer1, a_buffer2, batch_param_table, iteration, power_threshold, fourier_mask,M,No,Np)
--   -- print('P_F_without_background ' .. self.i)
--   local abs = zk_real_buffer
--   local da = a_buffer1
--   local fdev = a_buffer2
--   local err_fmag, renorm  = 0, 0
--   local batch_start, batch_end, batch_size = table.unpack(batch_param_table)
--   local module_error, mod_updates = 0, 0
--   -- u.printf('z norm: %g',z:normall(2)^2)
--   for i=11,13 do
--     -- plt:plot(z[i][1][1],'before exitwave '..i)
--     -- plt:plotcompare({z[i][1][1]:clone():float():log(),a[i]:clone():fftshift():float():log()},{'a_model','a'})
--   end
--   -- plt:plot(z[36][1][1],'before exitwave '..36)
--   z:view_3D():fftBatched()
--   -- plt:plot(z[36][1][1],'after  exitwave '..36)
--   -- u.printf('z norm: %g',z:normall(2)^2)
--   for k=1,batch_size do
--     local k_all = batch_start+k-1
--     -- sum over probe and object modes -> 1x1xMxM
--     local a_model = abs:normZ(z[k]):sum(1):sum(2)
--     a_model:sqrt()
--     -- pprint(abs)
--     -- pprint(a[k_all])
--     fdev[1][1]:add(a_model[1][1],-1,a[k_all])
--     if k_all % 11 == 0 then
--     -- self,imgs, title, suptitle, savepath
--       local path = '/mnt/f5c0a7bc-a539-461c-bc97-ed4eb92c48a1/Dropbox/Philipp/experiments/2017-24-01 monash/carbon_black/4000e/scan289/'
--       -- plt:plot(fdev[1][1]:clone():cdiv(a[k_all]):fftshift():float(),'fdev relative',path .. string.format('%d_it%d_fdev',k_all,i),false)
--       -- plt:plotcompare({a_model[1][1]:clone():fftshift():float(),a[k_all]:clone():fftshift():float()},{'a_model '..k_all,'a '..k_all},'',path .. string.format('%d_it%d_a',k_all,i),false)
--     end
--     da:pow(fdev,2)
--     da:cmul(fm[k_all])
--     err_fmag = da:sum()
--     -- u.printf('err_fmag = %g',err_fmag)
--     module_error = module_error + err_fmag
--
--     local fdev_exp =  fdev[1][1]:view(1,1,M,M):expand(No,Np,M,M)
--     local a_model_exp =  a_model[1][1]:view(1,1,M,M):expand(No,Np,M,M)
--
--     if err_fmag > power_threshold then
--       renorm = math.sqrt(power_threshold/err_fmag)
--       if fourier_mask then
--         fourier_mask:cmul(fm_exp[k_all])
--       else
--         fourier_mask = fm_exp[k_all]
--       end
--       -- m.P_Mod(z[k],a_model_exp,a_exp[k_all])
--       -- pprint(fourier_mask)
--       -- plt:plot(z[k][1][1],'before P_Mod_renorm '..k)
--       m.P_Mod_renorm(z[k],fourier_mask,fdev_exp,a_exp[k_all],a_model_exp,renorm)
--       -- plt:plot(z[k][1][1],'after  P_Mod_renorm '..k)
--       mod_updates = mod_updates + 1
--     end
--   end
--   z:view_3D():ifftBatched()
--   for i=11,13 do
--     -- plt:plot(z[i][1][1],'after exitwave '..i)
--     -- plt:plotcompare({z[i][1][1]:clone():float():log(),a[i]:clone():fftshift():float():log()},{'a_model','a'})
--   end
--   -- u.printf('z norm: %g',z:normall(2)^2)
--   i = i + 1
--   return module_error, mod_updates
-- end


return m
