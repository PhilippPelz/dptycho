local classic = require 'classic'
local u = require "dptycho.util"
local fn = require 'fn'
local py = require('fb.python')
local plot = require 'dptycho.io.plot'
local plt = plot()
local m = classic.module(...)

-- py.exec([=[
-- from math import *
-- from matplotlib import pyplot as plt
-- import numpy as np
-- from numpy.fft import fft, fft2, fftshift
-- from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NaviToolbar
-- from mpl_toolkits.axes_grid1 import make_axes_locatable
-- from scipy.sparse import csr_matrix
-- import scipy.sparse.linalg as linalg
-- import scipy.sparse as sp
-- import sys
-- import psutil
-- from numpy import linalg as LA
--
-- # Print iterations progress
-- def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
--     """
--     Call in a loop to create terminal progress bar
--     @params:
--         iteration   - Required  : current iteration (Int)
--         total       - Required  : total iterations (Int)
--         prefix      - Optional  : prefix string (Str)
--         suffix      - Optional  : suffix string (Str)
--         decimals    - Optional  : positive number of decimals in percent complete (Int)
--         barLength   - Optional  : character length of bar (Int)
--     """
--     formatStr       = "{0:." + str(decimals) + "f}"
--     percents        = formatStr.format(100 * (iteration / float(total)))
--     filledLength    = int(round(barLength * iteration / float(total)))
--     bar             = '█' * filledLength + '-' * (barLength - filledLength)
--     sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
--     sys.stdout.flush()
--     if iteration == total:
--         sys.stdout.write('\n')
--         sys.stdout.flush()
-- def applot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
--     im1, im2 = np.abs(img), np.angle(img)
--     fig, (ax1,ax2) = plt.subplots(1,2)
--     div1 = make_axes_locatable(ax1)
--     div2 = make_axes_locatable(ax2)
--     fig.suptitle(suptitle, fontsize=20)
--     imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
--     imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
--     cax1 = div1.append_axes("right", size="10%", pad=0.05)
--     cax2 = div2.append_axes("right", size="10%", pad=0.05)
--     cbar1 = plt.colorbar(imax1, cax=cax1)
--     cbar2 = plt.colorbar(imax2, cax=cax2)
--     ax1.set_title(title[0])
--     ax2.set_title(title[1])
--     plt.tight_layout()
--     if show:
--         plt.show()
--     if savePath is not None:
--         # print 'saving'
--         fig.savefig(savePath + '.png', dpi=300)
-- def riplot(img, suptitle='Image', savePath=None, cmap=['hot','hot'], title=['Abs','Phase'], show=True):
--     im1, im2 = np.real(img), np.imag(img)
--     fig, (ax1,ax2) = plt.subplots(1,2)
--     div1 = make_axes_locatable(ax1)
--     div2 = make_axes_locatable(ax2)
--     fig.suptitle(suptitle, fontsize=20)
--     imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
--     imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
--     cax1 = div1.append_axes("right", size="10%", pad=0.05)
--     cax2 = div2.append_axes("right", size="10%", pad=0.05)
--     cbar1 = plt.colorbar(imax1, cax=cax1)
--     cbar2 = plt.colorbar(imax2, cax=cax2)
--     ax1.set_title(title[0])
--     ax2.set_title(title[1])
--     plt.tight_layout()
--     if show:
--         plt.show()
--     if savePath is not None:
--         # print 'saving'
--         fig.savefig(savePath + '.png', dpi=300)
--
-- class DFT2d():
--     def __init__(self,N):
--         i, j = np.ogrid[0:N,0:N]
--         omega = np.exp( - 2 * pi * 1J / N )
--     #        W2 = np.power( omega, i * j ) / sqrt(N)
--         self.W2 = np.power( omega, i * j ) / sqrt(N)
--         self.N = N
--
--     def row(self,row):
-- #        applot(self.W2,'self.W2')
--         self.rowrep = np.broadcast_to(self.W2[row % self.N],(self.N,self.N))
-- #        applot(self.rowrep,'self.rowrep')
-- #        applot(self.W2[row/self.N].reshape((self.N,1)),'self.W2[row/self.N].reshape((self.N,1))')
--         self.res = self.rowrep * self.W2[row/self.N].reshape((self.N,1))
-- #        applot(self.res,'self.res')
--         return self.res.flatten()
--
--     def row_mul_diag(self,row, diag):
--         self.rowrep = np.broadcast_to(self.W2[row % self.N],(self.N,self.N))
--         self.res = self.rowrep * self.W2[row/self.N].reshape((self.N,1))
--         #applot(self.res.reshape((self.N,self.N)),'res')
--         self.res = self.res.flatten() * diag
--         #applot(self.res.reshape((self.N,self.N)),'res')
--         return self.res
--
-- # return the result of left multiplication of the 2d dft matrix with a diagonal masking matrix
-- def DFT2d_leftmasked(mask):
--     N = int(sqrt(mask.size))
--     s = mask.sum()
--     print s
--     print s*mask.size
--     row = np.ndarray((s*mask.size,),dtype=np.int)
--     col = np.ndarray((s*mask.size,),dtype=np.int)
--     data = np.ndarray((s*mask.size,),dtype=np.complex64)
--     dft = DFT2d(N)
--     ones = np.ones(N**2).astype(np.int)
--     cols = np.arange(N**2).astype(np.int)
--     col[:] = np.tile(cols,s)
--     i = 0
--     for r,m in enumerate(mask):
--         if m == 1:
-- #            print psutil.virtual_memory().used
--             row[i*N**2:(i+1)*N**2] = ones * r
-- #            print psutil.virtual_memory().used
--             data[i*N**2:(i+1)*N**2] = dft.row(r)
-- #            print psutil.virtual_memory().used
--             i += 1
-- #            gc.collect()
--             printProgress(i,s)
-- #            print r
--     print psutil.virtual_memory().used
--     return csr_matrix((data, (row, col)), shape=(N**2,N**2))
--
-- def DFT2d_leftmasked_mul_diag(mask,diag):
--     N = int(sqrt(mask.size))
--     s = mask.sum()
--     #print s
--     #print s*mask.size
--     row = np.ndarray((s*mask.size,),dtype=np.int)
--     col = np.ndarray((s*mask.size,),dtype=np.int)
--     data = np.ndarray((s*mask.size,),dtype=np.complex64)
--
--     dft = DFT2d(N)
--
--     ones = np.ones(N**2).astype(np.int)
--     cols = np.arange(N**2).astype(np.int)
--     col[:] = np.tile(cols,s)
--     i = 0
--     for r,m in enumerate(mask):
--         if m == 1:
-- #            print psutil.virtual_memory().used
--             row[i*N**2:(i+1)*N**2] = ones * r
-- #            print psutil.virtual_memory().used
--             data[i*N**2:(i+1)*N**2] = dft.row_mul_diag(r,diag)
-- #            print psutil.virtual_memory().used
--             i += 1
-- #            gc.collect()
--             printProgress(i,s)
-- #            print r
--     #print psutil.virtual_memory().used
--     return csr_matrix((data, (row, col)), shape=(N**2,N**2))
--
-- def largest_evec_Ta_Fv(Ta,v):
--     N = int(sqrt(Ta.size))
--     assert v.size == N**2
--     TaF = DFT2d_leftmasked(Ta)
--     #applot(TaF.toarray(),'TaF')
--     print psutil.virtual_memory().used
--     TaFv = TaF.dot(sp.diags(v,0,(N**2,N**2)))
--     print psutil.virtual_memory().used
--     #riplot(TaFv.toarray(),'TaFv')
--     try:
--         evals, evec = linalg.eigs(TaFv.toarray(),1)
--     except ArpackNoConvergence as anc:
--         print anc.message
--         print 'Eigenvector search did not converge'
--     except ArpackError as ae:
--         print ae.message
--         print 'Arpack Error'
--     print psutil.virtual_memory().used
--     print evals.real[0]
--     print evals.imag[0]
--     #applot(evec.reshape((N,N)),'evec')
--     return evals.real[0], evals.imag[0], np.abs(evec), np.angle(evec)
-- #    return 1,2,3,4
--
-- def largest_evec_Ta_Fv2(Ta,v):
--     N = int(sqrt(Ta.size))
--     assert v.size == N**2
--     TaFv = DFT2d_leftmasked_mul_diag(Ta,v)
--     #print psutil.virtual_memory().used
--     #riplot(TaFv.toarray(),'TaFv')
--     try:
--         evals, evec = linalg.eigs(TaFv,1,tol=1e-6)
--     except ArpackNoConvergence:
--         print 'Eigenvector search did not converge'
--     except ArpackError:
--         print 'Arpack Error'
--     #print psutil.virtual_memory().used
--     #print evals.real[0]
--     #print evals.imag[0]
--     #applot(evec.reshape((N,N)),'evec')
--     return evals.real[0], evals.imag[0], np.abs(evec), np.angle(evec)
--
-- def largest_evec_Ta_Fv_reals(Ta,vreal,vimag):
--     v = vreal + 1j* vimag
--     return largest_evec_Ta_Fv(Ta,v)
--
-- def largest_evec_Ta_Fv_reals2(Ta,vreal,vimag):
--     v = vreal + 1j* vimag
--     return largest_evec_Ta_Fv2(Ta,v)
--
-- ]=])

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
        -- pprint(z)
        -- pprint(z_h)
        -- pprint(key)
        -- pprint(k)
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
        zT_a_h = torch.FloatTensor(zT_a:size())
    else
        zT_a_h = torch.Tensor()
    end

    local partial_maybe_copy_new_batch = fn.partial(maybe_copy_new_batch,zT_a,zT_a_h,'z')

    -- preparations done

    local a_max = u.percentile(a:float(),truncation_threshold)
    local T_a = a_buffer:gt(a,a_max)

    for i=1,3 do
      plt:plotcompare({a[i]:float():log(), T_a[i]:float()},{'a',string.format('Ta[%d]',1)})
    end

    local ph = P:clone():view_3D():fftBatched()
    -- pprint(ph)
    plt:plot(ph[1]:zfloat(),string.format('fft p[%d]',1))
    ph = ph:arg():view(1,1,Np,M,M):expand(K,1,Np,M,M)

    local T_aCx = torch.ZCudaTensor(T_a:size()):polar(T_a,ph)
    plt:plot(T_aCx[1]:zfloat(),string.format('T_aCx[%d]',1))
    T_aCx:view_3D():ifftBatched()
    -- plt:plotReIm(T_aCx[1]:zfloat(),string.format('T_aCx[%d]',1))
    local T_a_exp = T_aCx:view(K,1,1,M,M):expand(K,No,Np,M,M)
    local Q_star_F_star_T_a = O_buffer
    local Q_star_F_star_T_a_views = ops.create_views(Q_star_F_star_T_a,pos,M)
    ops.Q_star(T_a_exp,P,Q_star_F_star_T_a,Q_star_F_star_T_a_views,zk_buffer,P_buffer,0,k_to_batch_index,partial_maybe_copy_new_batch,batches,K,dpos)
    -- plt:plot(O_denom[1][1]:float(),'O_denom')
    -- plt:plot(Q_star_F_star_T_a[1][1]:zfloat(),'Q_star_F_star_T_a')
    Q_star_F_star_T_a:cmul(O_denom)
    plt:plot(Q_star_F_star_T_a[1][1]:zfloat(),'Q_star_F_star_T_a norm')
    ops.Q(zT_a, P, Q_star_F_star_T_a_views, zk_buffer, k_to_batch_index,partial_maybe_copy_new_batch, batches, K,dpos)
    zT_a:view_3D():fftBatched()
    for k = 1,K do
        for no = 1, No do
            for np = 1, Np do
                -- plt:plot(zT_a[k][no][np]:zfloat(),'zT_a[k][no][np]')
                local v = zT_a[k][no][np]:zfloat():view(zT_a[k][no][np]:nElement())
                local m = T_a[k]:float():view(T_a[k]:nElement())
                -- pprint(v)
                -- pprint(m)
                local evalre, evalim, evec_abs, evec_phase = table.unpack(py.eval('largest_evec_Ta_Fv_reals2(Ta,vr,vi)',{Ta=m,vr=v:re(),vi=v:im()}))
                -- pprint(evalre)
                -- pprint(evalim)
                -- u.printf('eval = %g + %g i',evalre,evalim)
                -- z[k][no][np]:polar(1,zT_a[k][no][np]:cmul(T_a[k]):arg())
                z[k][no][np]:polar(evec_abs:cuda(),evec_phase:cuda())
                -- plt:plot(z[k][no][np]:zfloat(),'z[k][no][np]')
            end
        end
    end

    return z
end

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
function m.truncated_spectral_estimate_power_it(z,P,O_denom,truncation_threshold,ops,a,z_buffer, a_buffer,zk_buffer,P_buffer,O_buffer,batch_params,old_batch_params,k_to_batch_index,batches,batch_size,K,M,No,Np,pos,dpos,O_mask)

    local same_batch = function(batch_params1,batch_params2)
        local batch_start1, batch_end1, batch_size1 = table.unpack(batch_params1)
        local batch_start2, batch_end2, batch_size2 = table.unpack(batch_params2)
        return batch_start1 == batch_start2 and batch_end1 == batch_end2 and batch_size1 == batch_size2
    end

    local maybe_copy_new_batch = function (z,z_h,key,k)
        pprint(z)
        pprint(z_h)
        pprint(key)
        pprint(k)
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
    local zT_a_h = nil
    local partial_maybe_copy_new_batch = nil
    -- print('batches',batches)
    if batches > 1 then
        zT_a_h = torch.FloatTensor(zT_a:size())
        partial_maybe_copy_new_batch = fn.partial(maybe_copy_new_batch,zT_a,zT_a_h,'z')
    else
        zT_a_h = torch.Tensor()
        partial_maybe_copy_new_batch = function(k) end
    end

    -- local PI = P:clone():view_3D():fftBatched():norm():sum(1):re():expandAs(a)
    -- local a_diff = PI:add(-1,a):abs()
    --
    -- for i=20,30 do
    --   plt:plot(a_diff[i]:float(),'a_diff['..i..']')
    -- end
    -- print('wp 1')
    -- collectgarbage('collect')

    local O_views = ops.create_views(O_buffer,pos,M)

    local A = function()
      -- print('wp 21')
      -- collectgarbage('collect')
      -- plt:plot(P[1][1]:zfloat(),string.format('z1 %d',1))
      z_buffer = ops.Q(z_buffer, P, O_views, zk_buffer, k_to_batch_index,partial_maybe_copy_new_batch, batches, K,dpos)
      -- print('wp 22')
      -- pprint(z_buffer)
      -- plt:plot(z_buffer[1][1][1]:zfloat(),string.format('z1 %d',1))
      -- plt:plot(z_buffer[20][1][1]:zfloat(),string.format('z20 %d',1))
      z_buffer:view_3D():fftBatched()
      -- print('wp 23')
      -- plt:plot(z_buffer[1][1][1]:zfloat(),string.format('z1 %d',2))
      -- plt:plot(z_buffer[20][1][1]:zfloat(),string.format('z20 %d',2))
      return z_buffer
    end

    local At = function(z)
      z:view_3D():ifftBatched()
      -- plt:plot(z_buffer[20][1][1]:zfloat(),string.format('z20 %d',1))
      ops.Q_star(z,P,O_buffer,O_views,zk_buffer,P_buffer,0,k_to_batch_index,partial_maybe_copy_new_batch,batches,K,dpos)
      O_buffer:cmul(O_denom)
      return O_buffer
    end

    local t = torch.randn(O_buffer:nElement()*2):float()
    local zt = torch.ZFloatTensor(t:resize(O_buffer:nElement(),2))
    O_buffer:copy(zt)
    -- plt:plot(O_buffer[1][1]/:zfloat(),string.format('O %d',1))

    local normest = math.sqrt(a:sum()/a:nElement())
    -- print(9*normest^2)
    local a_max = u.percentile(a:float(),truncation_threshold)
    local T_a = a_buffer:gt(a,a_max)
    -- local T_a = a_buffer:le(a,15*normest^2)
    -- for i=20,30 do
    --   plt:plotcompare({a[i]:float(), T_a[i]:float()},{'a',string.format('Ta[%d]',1)})
    -- end
    local T_a_exp = T_a:cmul(a):view(K,1,1,M,M):expand(K,No,Np,M,M)

    -- for i=20,30 do
    --   plt:plotcompare({a[i]:float(), T_a[i]:float()},{'a',string.format('Ta[%d]',1)})
    -- end
    -- :view(K,1,1,M,M):expand(K,No,Np,M,M)
    -- print('wp 2')
    local O = O_buffer
    for tt = 1,70 do
      local z = A()
      -- print('wp 3 '..tt)
      -- plt:plot(z[1][1][1]:zfloat(),string.format('z1 %d',tt))
      -- plt:plot(z[20][1][1]:zfloat(),string.format('z20 before %d',tt))
      z:cmul(T_a_exp)
      -- plt:plot(z[1][1][1]:zfloat(),string.format('z1 %d',tt))
      -- plt:plot(z[20][1][1]:zfloat(),string.format('z20 after  %d',tt))
      O = At(z)
      -- print('wp 4 '..tt)
      O:div(O:normall(2))
      -- if tt % 200 == 0 then
      plt:plot(O[1][1]:zfloat(),string.format('O %d',tt))
      -- end
    end

    O_buffer:mul(normest)
    -- plt:plot(O_buffer[1][1]:zfloat(),'O_buffer')
    return O_buffer
end


return m
