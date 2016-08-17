local classic = require 'classic'
local u = require "dptycho.util"
local znn = require 'dptycho.znn'
local zt = require "ztorch.fcomplex"
local xlua = require "xlua"

local m = classic.class(...)
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
function m.static.calculateO_denom(O_denom,O_mask,O_denom_views,P,P_buffer_real,Pk_buffer_real,O_inertia,K,O_denom_regul_factor_start,O_denom_regul_factor_end,i,it,dpos)
  if O_inertia then
    O_denom:fill(O_inertia*K)
  else
    O_denom:fill(0)
  end

  -- 1 x 1 x M x M
  local norm_P_shifted = Pk_buffer_real
  -- 1 x 1 x M x M
  P_buffer_real:normZ(P):sum(2)
  local norm_P = P_buffer_real[{{1},{1},{},{}}]
  -- plt:plot(self.P[1][1]:zfloat(),'self.P 0')
  -- plt:plot(norm_P[1][1]:float(),'self.P_buffer_real 0')
  -- 1 x Np x M x M
  local tmp = P_buffer_real
  for k,view in ipairs(O_denom_views) do
    norm_P_shifted[1]:shift(norm_P[1],dpos[k])
    local np_exp = norm_P_shifted:expandAs(view)
    -- plt:plot(np_exp[1][1]:float(),'np_exp')
    view:add(np_exp)
  end
  local abs_max = tmp:absZ(P):max()
  local fact = O_denom_regul_factor_start-(i/it)*(O_denom_regul_factor_start-O_denom_regul_factor_end)
  local sigma =  abs_max * abs_max * fact
  -- u.printf('sigma, fact, abs = %g, %g, %g',sigma,fact,abs_max)
  -- plt:plot(self.O_denom[1][1]:float(),'self.O_denom')
  m.InvSigma(O_denom,sigma)
  O_mask:lt(O_denom,1e-5)
  u.printram('after calculateO_denom')
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
RETURN:
- `z`     : the new frames
]]
function m.static.Q(z, mul_merge, merge_memory, merge_memory_views, zk_buffer, P_buffer,O_inertia, k_to_batch_index, batch_copy_func,batches,K,dpos)
    local product_shifted = zk_buffer
    local mul_merge_shifted = P_buffer

    if O_inertia then
      merge_memory:fill(O_inertia*K)
    else
      merge_memory:fill(0)
    end

    -- plt:plot(mul_merge[1][1]:zfloat(),'mul_merge')
    local pos = torch.FloatTensor{1,1}
    u.printram('before merge_frames')

    for k, view in ipairs(merge_memory_views) do
      if batches > 2 then xlua.progress(k,K) end
      pos:fill(1):cmul(dpos[k])
      batch_copy_func(k)
      local ind = k_to_batch_index[k]
      mul_merge_shifted[1]:shift(mul_merge[1],pos)
      mul_merge_shifted[1]:conj()
      local mul_merge_shifted_expanded = mul_merge_shifted:expandAs(z[ind])

      -- z * P^*
      product_shifted:cmul(z[ind],mul_merge_shifted_expanded):sum(2)
      -- plt:plot(product_shifted[1][1]:zfloat(),'product_shifted')
      view:add(product_shifted[1][1])
      -- plt:plot(merge_memory[1][1]:zfloat(),'merge_memory')
    end
    -- plt:plot(merge_memory[1][1]:zfloat(),'merge_memory')
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
function m.static.Q_star(z,mul_split,merge_memory_views,zk_buffer,k_to_batch_index,batch_copy_func,batches,K,dpos)
  local mul_split_shifted = zk_buffer
  local pos = torch.FloatTensor{1,1}
  for k, view in ipairs(merge_memory_views) do
    if batches > 2 then xlua.progress(k,K) end
    batch_copy_func(k)
    local ind = k_to_batch_index[k]
    -- print(k,ind)
    pos:fill(1):cmul(dpos[k])
    mul_split_shifted:view_3D():shift(mul_split:view_3D(),pos)
    local view_exp = view:expandAs(z[ind])
    -- plt:plot(mul_split_shifted[1][1]:zfloat(),'mul_split_shifted')
    -- plt:plot(view_exp[1][1]:zfloat(),'view_exp')
    z[ind]:cmul(mul_split_shifted,view_exp)
    -- u.printf('||z|| = %g',z[ind]:normall(2)^2)
  end
  -- plt:plot(z[1][1][1]:zfloat(),'z[1]')
  u.printram('after update_frames')
end

function m.static.refine_probe(z,P,O_views,P_buffer1,P_buffer_real1,P_buffer_real2,zk_buffer1,zk_buffer2,zk_buffer_real1,k_to_batch_index,batch_copy_func,dpos,probe_support,P_inertia)
  local new_P = P_buffer1
  local dP = P_buffer1

  local new_P_denom = P_buffer_real1
  local denom_shifted = P_buffer_real2

  local oview_conj = zk_buffer1
  local oview_conj_shifted = zk_buffer2

  local denom_tmp = zk_buffer_real1

  if P_inertia then
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
    pos:fill(-1):cmul(dpos[k])
    oview_conj:conj(view:expandAs(z[ind]))

    denom_tmp = denom_tmp:normZ(oview_conj):sum(1)
    denom_shifted[1]:shift(denom_tmp[1],pos)

    oview_conj:cmul(z[ind])
    oview_conj_shifted:view_3D():shift(oview_conj:view_3D(),pos)

    new_P_denom:add(denom_shifted)
    new_P:add(oview_conj_shifted:sum(1))
  end
  new_P:cdiv(new_P_denom)

  P:copy(new_P)
  if probe_support then P = probe_support:forward(P) end

  -- plt:plot(self.P[1]:zfloat(),'self.P')
  -- if self.probe_regularization_amplitude(self.i) then self:regularize_probe() end
  -- if self.probe_lowpass_fwhm(self.i) then self:filter_probe() end

  u.printram('after refine_probe')
  return math.sqrt(dP:add(new_P,-1,P):normall(2)^2/P:normall(2)^2)
end


return m
