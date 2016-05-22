local classic = require 'classic'
local base_engine = require "dptycho.core.ptycho.base_engine_shifted"
local u = require 'dptycho.util'
local plot = require 'dptycho.io.plot'
local TruncatedPoissonLikelihood = require 'dptycho.znn.TruncatedPoissonLikelihood'
local plt = plot()
local engine, super = classic.class(...,base_engine)

function engine:_init(par, a_h)
  super._init(self,par)
  self.a_h = a_h
end

function engine:allocateBuffers(K,No,Np,M,Nx,Ny)
  
  self.L = TruncatedPoissonLikelihood(self.a_h, self.gradInput, self.fm, buffer1, buffer2, self.par, K, No, Np)
end

function engine:iterate(steps)

end
