local classic = require 'classic'
local u = require "dptycho.util"
local znn = require 'dptycho.znn'
local zt = require "ztorch.fcomplex"
local xlua = require "xlua"
local plot = require 'dptycho.io.plot'
local plt = plot()
local m = classic.class(...)
local cutorch = require 'cutorch'
m:include('dptycho.core.ptycho.ops_general')

--[[ calculate the operator (Q^star Q)^-1
ARGS:
- 'O_denom'           : the frames to split into, el CC [K,No,Np,M,M]
- 'O_mask`            :
- 'O_denom_views'     :
- `P`                 :
- 'P_buffer_real'     :
- 'O_inertia'         :
- 'K'                 :
- 'O_denom_regul_factor_start'      :
- 'O_denom_regul_factor_end'        :
- 'i'                               :
- 'it'                              :
RETURN:
- `z`     : the new frames
]]
function m.static.calculateO_denom(O_denom,O_mask,O_denom_views,P,P_buffer,Pk_buffer_real,O_inertia,K,O_denom_regul_factor_start,O_denom_regul_factor_end,i,it,dpos)
  P_buffer:zero()
  local P_dim = 2
  if O_inertia ~= 0 then
    O_denom:fill(O_inertia)
  else
    O_denom:fill(1e-3)
  end
  -- local P_norm = P_buffer:norm(P):sum(P_dim)
  -- for k,view in ipairs(O_denom_views) do
  --   Pk_buffer_real[1]:shift(P_norm[1]:re(),dpos[k])
  --   local np_exp = Pk_buffer_real:expandAs(view)
  --   -- plt:plot(np_exp[1][1]:float(),'np_exp')
  --   view:add(np_exp)
  --   -- plt:plot(O_denom[1][1]:float(),'self.O_denom')
  -- end
  for k,view in ipairs(O_denom_views) do
    P_buffer[1]:shift(P[1],dpos[k])
    -- 1 x 1 x M x M
    local norm_P_shifted = Pk_buffer_real:normZ(P_buffer)
    norm_P_shifted = norm_P_shifted:sum(P_dim)
    local np_exp = norm_P_shifted:expandAs(view)
    -- plt:plot(np_exp[1][1]:float(),'np_exp')
    view:add(np_exp)
    -- plt:plot(O_denom[1][1]:float(),'self.O_denom')
  end
  local abs_max = P_buffer:abs(P):max()
  local fact = O_denom_regul_factor_start-(i/it)*(O_denom_regul_factor_start-O_denom_regul_factor_end)
  local sigma =  abs_max * abs_max * fact
  -- u.printf('sigma, fact, abs = %g, %g, %g',sigma,fact,abs_max)

  local n = O_denom:clone():div(O_denom:max())
  if O_mask then
    O_mask:gt(n,6e-2)
    -- plt:plot(O_mask[1][1]:float(),'O_mask')
  end
  m.InvSigma(O_denom,sigma)
  u.printram('after calculateO_denom')
  -- plt:plot(O_denom[1][1]:float(),'self.O_denom')
  return O_denom, O_mask
end

--[[ merge all frames into the object
ARGS:
- 'z'           : the frames to merge together, el CC [K,No,Np,M,M]
- 'O_denom'     : the operator (Q^star Q)^-1
- 'mul_merge`   :
- 'merge_memory'          :
- `merge_memory_views`    :
- 'product_shifted_buffer':
- 'zk_buffer'             : [No,Np,M,M]
- 'P_buffer'              : [1 ,Np,M,M]
- 'O_inertia'             :
- 'k_to_batch_index'          :
- 'batch_copy_func'           :
- 'batches'           :
- 'K'           :
- 'dpos'           :
RETURN:
- `z`     : the new frames
]]
function m.static.Q_star(z, mul_merge, merge_memory, merge_memory_views, zk_buffer, P_buffer,O_inertia, k_to_batch_index, batch_copy_func,batches,K,dpos,O_filter)
    local product_shifted = zk_buffer
    local mul_merge_shifted = P_buffer
    -- print(O_inertia)
    -- merge_memory = merge_memory:clone()
    if O_inertia then
      if O_filter then
        print('filtering object')
        m.filter_object(merge_memory,O_filter)
      end
      merge_memory:mul(O_inertia)
    else
      merge_memory:fillIm(0):fillRe(0)
    end
    -- print('merge_memory max min',merge_memory:re():max(),merge_memory:im():max())
    -- plt:plotReIm(merge_memory[1][1]:zfloat(),'merge_memory Q_star beginning')
    u.printram('before merge_frames')
    for k, view in ipairs(merge_memory_views) do
      if batches > 2 then xlua.progress(k,K) end
      batch_copy_func(k)
      local ind = k_to_batch_index[k]
      mul_merge_shifted[1]:shift(mul_merge[1],dpos[ind],zk_buffer[1][1])
      mul_merge_shifted:conj()
      -- z * P^*
      local zP = product_shifted:cmul(z[ind],mul_merge_shifted:expandAs(z[ind])):sum(2)
      if k > 86 and k < 90 then
        -- plt:plot(z[ind][1][1]:zfloat(),'z[ind][1][1]')
        -- plt:plot(mul_merge_shifted[1][1]:zfloat(),'mul_merge_shifted_expanded')
        -- plt:plot(zP[1][1]:zfloat(),'product_shifted')
        -- plt:plot(merge_memory[1][1]:zfloat(),'merge_memory')
      end
      view:add(zP)
      -- plt:plotReIm(view[1][1]:zfloat(),'view')
      -- plt:plotReIm(merge_memory[1][1]:zfloat(),'merge_memory '..k)
    end
    -- plt:plot(merge_memory[1][1]:zfloat(),'merge_memory')
    -- local f = hdf5.open(path.. 'object/merged.h5')
    -- local options = hdf5.DataSetOptions()
    -- options:setChunked(1,1,64, 64)
    -- options:setDeflate(8)
    -- f:write('/obre',merge_memory[1][1]:zfloat():re())
    -- f:write('/obim',merge_memory[1][1]:zfloat():im())
    -- f:close()

    u.printram('after merge_frames')
end

--[[ split the object into frames
ARGS:
- 'z'           : the frames to split into, el CC [K,No,Np,M,M]
- 'mul_split`   :
- 'merge_memory_views'          :
- `batch_copy_func`    :
- 'k_to_batch_index':
- 'batches'             :
- 'K'             :
- 'dpos'             :
RETURN:
- `z`     : the new frames
]]
function m.static.Q(z,mul_split,merge_memory_views,zk_buffer,shift_buffer,k_to_batch_index,batch_copy_func,batches,K,dpos)
  local mul_split_shifted = zk_buffer
  for k, view in ipairs(merge_memory_views) do
    if batches > 2 then xlua.progress(k,K) end
    batch_copy_func(k)
    local ind = k_to_batch_index[k]
    mul_split_shifted[1]:shift(mul_split[1],dpos[ind],shift_buffer)
    local view_exp = view:expandAs(z[ind])
    for i = 2, mul_split_shifted:size(1) do
      mul_split_shifted[i]:copy(mul_split_shifted[1])
    end
    if ind > 80 and ind < 100 then
      -- plt:plot(mul_split_shifted[1][1]:zfloat(),'mul_split_shifted')
      -- plt:plot(view_exp[1][1]:zfloat(),'view_exp')
    end
    z[ind]:cmul(mul_split_shifted,view_exp)
    -- u.printf('||z|| = %g',z[ind]:normall(2)^2)
  end
  -- plt:plot(z[1][1][1]:zfloat(),'z[1]')
  u.printram('after update_frames')
  return z
end

function m.static.refine_probe(z,P,O_views,P_buffer1,P_buffer2,P_buffer_real1,P_buffer_real2,zk_buffer1,zk_buffer2,zk_buffer_real1,k_to_batch_index,batch_copy_func,dpos,probe_support,P_inertia,P_filter)
  -- local O_dim = 1
  -- local new_P = P_buffer1
  -- local dP = P_buffer2:zero()
  --
  -- local new_P_denom = P_buffer_real1
  -- local denom_shifted = P_buffer_real2
  --
  -- local oview_conj = zk_buffer1
  -- local oview_conj_shifted = zk_buffer2
  --
  -- local denom_tmp = zk_buffer2
  --
  -- if P_inertia then
  --   new_P:mul(P,P_inertia)
  --   new_P_denom:fill(P_inertia)
  -- else
  --   new_P:fill(1e-12)
  --   new_P_denom:fill(1e-12)
  -- end
  --
  -- local pos = torch.FloatTensor{1,1}
  -- for k, view in ipairs(O_views) do
  --   batch_copy_func(k)
  --   local ind = k_to_batch_index[k]
  --   pos:fill(-1):cmul(dpos[k])
  --   oview_conj:view_3D():conj(view[{{},{1},{},{}}])
  --   -- plt:plot(oview_conj[1][1]:zfloat(),'oview_conj')
  --
  --   denom_tmp:view_3D():shift(oview_conj:view_3D(),pos)
  --   -- plt:plot(denom_tmp[1][1]:zfloat(),'denom_tmp')
  --   denom_shifted[{{},{1},{},{}}]:normZ(denom_tmp[{{},{1},{},{}}]):sum(O_dim)
  --   -- plt:plot(denom_shifted[1][1]:float(),'denom_shifted')
  --   new_P_denom:add(denom_shifted)
  --   local nans4 = torch.ne(denom_shifted,denom_shifted)
  --   -- plt:plot(new_P_denom[1][1]:float(),'new_P_denom')
  --
  --   oview_conj = oview_conj:expandAs(z[ind])
  --   -- plt:plot(oview_conj[1][1]:zfloat(),'oview_conj')
  --   local nans7  = torch.ne(oview_conj:re(),oview_conj:re())
  --   local nans77  = torch.ne(oview_conj:im(),oview_conj:im())
  --   local nans8  = torch.ne(z[ind]:re(),z[ind]:re())
  --   local nans88  = torch.ne(z[ind]:im(),z[ind]:im())
  --   oview_conj:cmul(z[ind])
  --   -- plt:plot(oview_conj[1][1]:zfloat(),'oview_conj * z')
  --   local nans6  = torch.ne(oview_conj:re(),oview_conj:re())
  --   local nans66  = torch.ne(oview_conj:im(),oview_conj:im())
  --   oview_conj_shifted:view_3D():shift(oview_conj:view_3D(),pos)
  --   -- plt:plot(oview_conj_shifted[1][1]:zfloat(),'oview_conj_shifted * z')
  --   local nans5  = torch.ne(oview_conj_shifted:re(),oview_conj_shifted:re())
  --   local nans55  = torch.ne(oview_conj_shifted:im(),oview_conj_shifted:im())
  --   new_P:add(oview_conj_shifted:sum(O_dim))
  --   -- plt:plot(new_P[1][1]:zfloat(),'new_P')
  --   local nans1 = torch.ne(new_P:re(),new_P:re())
  --   local nans2 = torch.ne(new_P:im(),new_P:im())
  --   local nans3 = torch.ne(new_P_denom,new_P_denom)
  --   print(k,nans1:sum(),nans2:sum(),nans3:sum(),nans4:sum(),nans5:sum(),nans55:sum(),nans6:sum(),nans66:sum(),nans7:sum(),nans77:sum(),nans8:sum(),nans88:sum())
  --   print()
  -- end
  -- plt:plot(new_P_denom[1][1]:float(),'new_P_denom')
  -- plt:plot(new_P[1][1]:zfloat(),'new_P')
  -- local nans1 = torch.ne(new_P:re(),new_P:re())
  -- local nans2 = torch.ne(new_P:im(),new_P:im())
  -- local nans3 = torch.ne(new_P_denom,new_P_denom)
  -- print(nans1:sum(),nans2:sum(),nans3:sum())
  -- new_P:cdiv(new_P_denom)
  -- plt:plot(new_P[1][1]:zfloat(),'new_P:cdiv(new_P_denom)')
  -- dP:add(new_P,-1,P)
  -- print(dP:min(),dP:max())
  -- plt:plot(dP[1][1]:zfloat(),'dP')
  -- local p_change = dP:normall(2)^2
  -- local p_norm = P:normall(2)^2
  -- print(p_change,p_norm)
  -- P:copy(new_P)
  --
  -- if probe_support then P = probe_support:forward(P) end
  --
  -- -- plt:plot(P[1][1]:zfloat(),'self.P')
  -- -- if self.probe_regularization_amplitude(self.i) then self:regularize_probe() end
  -- -- if self.probe_lowpass_fwhm(self.i) then self:filter_probe() end
  --
  -- u.printram('after refine_probe')
  -- return math.sqrt(p_change/p_norm)
  local new_P = P_buffer1
  local dP = P_buffer2

  local new_P_denom = P_buffer_real1
  local denom_shifted = P_buffer_real2

  local oview_conj = zk_buffer1
  local oview_conj_shifted = zk_buffer2

  local denom_tmp = zk_buffer_real1

  if P_inertia then
    if P_filter then
      print('filtering probe')
      m.filter_probe(P,P_filter)
    end
    new_P:mul(P,P_inertia)
    new_P_denom:fill(P_inertia)
  else
    new_P:fill(0)
    new_P_denom:fill(1e-12)
  end

  local pos = torch.FloatTensor{1,1}
  for k, view in ipairs(O_views) do
    batch_copy_func(k)
    local ind = k_to_batch_index[k]
    denom_shifted:zero()
    pos:fill(-1):cmul(dpos[ind])
    oview_conj:conj(view:expandAs(z[ind]))

    oview_conj_shifted:view_3D():shift(oview_conj:view_3D(),pos,P_buffer2[1][1])
    local denom = denom_tmp:normZ(oview_conj_shifted):sum(1)

    oview_conj:cmul(z[ind])
    oview_conj_shifted:view_3D():shift(oview_conj:view_3D(),pos,P_buffer2[1][1])

    new_P_denom:add(denom)
    new_P:add(oview_conj_shifted:sum(1))
  end
  -- plt:plot(new_P_denom[1][1]:float():log(),'new_P_denom')
  -- plt:plot(new_P[1][1]:zfloat(),'new_P')
  new_P:cdiv(new_P_denom)
  -- plt:plot(new_P[1][1]:zfloat(),'new_P:cdiv(new_P_denom)')
  P:copy(new_P)
  if probe_support then P = probe_support:forward(P) end
  -- plt:plot(P[1][1]:zfloat(),'new_P:cdiv(new_P_denom)')
  -- plt:plot(self.P[1]:zfloat(),'self.P')
  -- if self.probe_regularization_amplitude(self.i) then self:regularize_probe() end
  -- if self.probe_lowpass_fwhm(self.i) then self:filter_probe() end

  u.printram('after refine_probe')
  return math.sqrt(dP:add(new_P,-1,P):normall(2)^2/P:normall(2)^2)
end


return m
