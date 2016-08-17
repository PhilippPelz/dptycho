local classic = require 'classic'
local u = require "dptycho.util"
local fn = require 'fn'

local m = classic.module(...)

--[[  truncated phase initialisation
      compute the phase of the largest EV of T_a F Q (Q^* Q)^-1 Q^* F^* T_a
      Ref:
      .Marchesini, S., Tu, Y. & Wu, H. Alternating Projection, Ptychographic Imaging and Phase Synchronization. arXiv Prepr. arXiv1402.0550 1–29 (2014). p. 27
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
function m.truncated_spectral_estimate(z,P,O_denom,truncation_threshold,ops,a,z_buffer, a_buffer,zk_buffer,P_buffer,O_buffer,batch_params,old_batch_params,k_to_batch_index,batches,batch_size,K,M,No,Np,pos,dpos)

    local same_batch = function(batch_params1,batch_params2)
        local batch_start1, batch_end1, batch_size1 = table.unpack(batch_params1)
        local batch_start2, batch_end2, batch_size2 = table.unpack(batch_params2)
        return batch_start1 == batch_start2 and batch_end1 == batch_end2 and batch_size1 == batch_size2
    end

    local maybe_copy_new_batch = function (z,z_h,key,k)
        if (k-1) % batch_size == 0 and batches > 1 then
            local batch = math.floor(k/batch_size) + 1

            local oldparams = old_batch_params['z']
            old_batch_params['z'] = batch_params[batch]
            local batch_start, batch_end, batch_size = table.unpack(batch_params[batch])
            u.debug('----------------------------------------------------------------------')
            -- u.debug('batch '..batch)
            u.debug('%s: s, e, size             = (%03d,%03d,%03d)',key,batch_start, batch_end, batch_size)

            if oldparams then
                local old_batch_start, old_batch_end, old_batch_size = table.unpack(oldparams)
                u.debug('%s: old_s, old_e, old_size = (%03d,%03d,%03d)',key,old_batch_start, old_batch_end, old_batch_size)
                if not same_batch(oldparams,batch_params[batch]) then
                    z_h[{{old_batch_start,old_batch_end},{},{},{},{}}]:copy(z[{{1,old_batch_size},{},{},{},{}}])
                    z[{{1,batch_size},{},{},{},{}}]:copy(z_h[{{batch_start,batch_end},{},{},{},{}}])
                end
            else
                z[{{1,batch_size},{},{},{},{}}]:copy(z_h[{{batch_start,batch_end},{},{},{},{}}])
            end
        end
        -- u.printram('after maybe_copy_new_batch')
    end


    local zT_a = z_buffer
    local F_PQ_Fstar_Ta = z

    local zT_a_h = nil
    if batches > 1 then
        local zT_a_h = torch.FloatTensor(zT_a:size())
    end
    local partial_maybe_copy_new_batch = fn.partial(maybe_copy_new_batch,zT_a,zT_a_h,'z')

    local F = u.DTF2D(M)

    -- preparations done

    local a_max = a:max()
    local T_a = a_buffer:gt(a,truncation_threshold*a_max)

    T_a:ifftBatched()
    local T_a_exp = T_a:view(K,1,1,M,M):expand(K,No,Np,M,M)

    local Q_star_F_star_T_a = O_buffer
    local Q_star_F_star_T_a_views = ops.create_views(Q_star_F_star_T_a,pos,M)

    ops.Q_star(T_a_exp,P,Q_star_F_star_T_a_views,zk_buffer,k_to_batch_index,partial_maybe_copy_new_batch,batches,K,dpos)

    -- plt:plotReIm(self.O[1][1]:zfloat(),'O after merge 0')
    Q_star_F_star_T_a:cmul(O_denom)

    ops.Q(zT_a, P, Q_star_F_star_T_a, Q_star_F_star_T_a_views, zk_buffer, P_buffer,0, k_to_batch_index,partial_maybe_copy_new_batch, batches, K,dpos)

    for k = 1,K do
        for no = 1, No do
            for np = 1, Np do
                local v = zT_a[k][no][np]:view(M^2,1)
                local m = torch.diag(T_a[k]:view(T_a[k]:nElement()))
                local Fv = torch.mm(F,v)
                local T_aFv = torch.mm(m,Fv)
                local e,v = torch.eig(T_aFv,'V')
                e = e[{{},{1}}]
                local max, imax = torch.max(e:abs())
                local vmax_phase = v[{{},{imax}}]:arg()
                z[k][no][np]:polar(1,vmax_phase)
            end
        end
    end

    return z
end




return m
