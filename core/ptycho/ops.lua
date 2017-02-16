local classic = require 'classic'
local u = require "dptycho.util"
local znn = require 'dptycho.znn'
local zt = require "ztorch.fcomplex"
local plot = require 'dptycho.io.plot'
local plt = plot()
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
  if O_inertia ~= 0 then
    O_denom:fill(O_inertia)
  else
    O_denom:fill(1e-12)
  end

  -- probe intensity
  local norm_P =  P_buffer_real:normZ(P):sum(2)
  norm_P = norm_P:expandAs(O_denom_views[1])
  for _, view in ipairs(O_denom_views) do
    view:add(norm_P)
  end
  local abs_max = P_buffer_real:absZ(P):max()
  local fact = O_denom_regul_factor_start-(i/it)*(O_denom_regul_factor_start-O_denom_regul_factor_end)
  local sigma =  abs_max * abs_max * fact
  local n = O_denom:clone():div(O_denom:max())
  O_mask:gt(n,1e-2)
  -- plt:plot(O_denom[1][1]:float(),'O_denom')
  m.InvSigma(O_denom,sigma)
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
- 'zk_buffer'             :
- 'O_inertia'             :
- 'do_normalize_merge_memory' :
- 'k_to_batch_index'          :
- 'batch_copy_func'           :
RETURN:
- `z`     : the new frames
]]
function m.static.Q_star(z, mul_merge, merge_memory, merge_memory_views, zk_buffer, P_buffer,O_inertia, k_to_batch_index, batch_copy_func,batches,K,dpos)
  local mul_merge_repeated = zk_buffer
  if O_inertia ~= 0 then
    merge_memory:mul(O_inertia)
  else
    merge_memory:fill(0)
  end
  -- plt:plotReIm(mul_merge[1][1]:zfloat(),'mul_merge')
  -- plt:plotReIm(merge_memory[1][1]:zfloat(),'merge_memory')
  for k, view in ipairs(merge_memory_views) do
    if batches > 2 then xlua.progress(k,K) end
    batch_copy_func(k)
    local ind = k_to_batch_index[k]
    mul_merge_repeated:conj(mul_merge:expandAs(z[ind]))
    -- plt:plotReIm(mul_merge_repeated[1][1]:zfloat(),'mul_merge_repeated '..k)
    -- plt:plotReIm(z[ind][1][1]:zfloat(),'z '..k)
    -- pprint(mul_merge_repeated)
    -- pprint(z[ind])
    local a = z[ind]:clone():cmul(mul_merge_repeated)
    -- plt:plotReIm(a[1][1]:zfloat(),'mergeme '..k)
    if a:size(2) > 1 then
      a = a:sum(2)
    end
    -- pprint(a)
    -- plt:plotReIm(a[1][1]:zfloat(),'mergeme '..k)
    -- add sum over probe modes
    view:add(a)
    -- view:add(mul_merge_repeated:fillIm(1):fillRe(1))
    if true then
      -- plt:plotReIm(merge_memory[1][1]:zfloat(),'merged '..k)
    end
  end
  -- plt:plotReIm(merge_memory[1][1]:zfloat(),'merged '..1)
  return z
end



--[[ split the object into frames
ARGS:
- 'z'                     : the frames to split into, el CC [K,No,Np,M,M]
- 'mul_split`             :
- 'merge_memory_views'    :
- `batch_copy_func`       :
- 'k_to_batch_index'      :
- 'engine'                :
RETURN:
- `z`                     : the new frames
]]
function m.static.Q(z,mul_split,merge_memory_views,zk_buffer,k_to_batch_index,batch_copy_func,batches,K,dpos)
  -- pprint(mul_split)
  -- pprint(merge_memory_views[1])
  -- pprint(z)
  for k, view in ipairs(merge_memory_views) do
    if batches > 2 then xlua.progress(k,K) end
    batch_copy_func(k)
    local ind = k_to_batch_index[k]
    z[ind]:cmul(view:expandAs(z[ind]),mul_split:expandAs(z[ind]))
    -- plt:plot(view[1][1]:zfloat(),'view[ind]')
    -- plt:plot(z[ind][1][1]:zfloat(),'z[ind]')
  end
  return z
end

function m.static.refine_probe(z,P,O_views,P_buffer1,P_buffer_real1,P_buffer_real2,zk_buffer1,zk_buffer2,zk_buffer_real1,k_to_batch_index,batch_copy_func,dpos,probe_support,P_inertia)
  local new_P = P_buffer1
  local dP = P_buffer1
  local dP_abs = P_buffer_real1
  local new_P_denom = P_buffer_real2

  local oview_conj = zk_buffer1
  local denom_tmp = zk_buffer_real1

  if P_inertia ~= 0 then
    new_P:mul(P,P_inertia)
    new_P_denom:fill(P_inertia)
  else
    new_P:fill(0)
    new_P_denom:fill(1e-12)
  end

  for k, view in ipairs(O_views) do
    batch_copy_func(k)
    local ind = k_to_batch_index[k]
    local view_exp = view:expandAs(z[ind])
    oview_conj:conj(view_exp)
    new_P:add(oview_conj:cmul(z[ind]):sum(1))
    new_P_denom:add(denom_tmp:normZ(view_exp):sum(1))
  end
  new_P:cdiv(new_P_denom)
  P:copy(new_P)
  if probe_support then
    P = probe_support:forward(P)
  end
  return math.sqrt(dP:add(new_P,-1,P):normall(2)^2/P:normall(2)^2)
end

return m
