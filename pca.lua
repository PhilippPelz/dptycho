require 'hdf5'
require 'torch'
require 'ztorch'
require 'zcutorch'
require 'unsup'
local pprint = require "pprint"
local plot = require 'dptycho.io.plot'
local builder = require 'dptycho.core.netbuilder'
local optim = require "optim"
local znn = require "dptycho.znn"
local plt = plot()

local M = 1536
local NP = 3
local path = '/home/philipp/experiments/2016-03-14 ptycho/scan/hyperscan_flipped_2/'
local file = 'ptycho_10.h5'
f = hdf5.open(path..file,'r')
local pr = f:read('/pr'):all():cuda()
local pi = f:read('/pi'):all():cuda()
-- pprint(pr)
-- pprint(pi)
local probe = torch.ZCudaTensor().new({1,NP,M,M})
probe:copyIm(pi):copyRe(pr)
probe:copyIm(pi):copyRe(pr)
f:close()
p = probe:squeeze()
pprint(p)
local pcol = p:view(NP,M*M)
pprint(pcol)
s,w = unsup.pca(pcol)
pprint(s)
pprint(w)
