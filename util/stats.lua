local classic = require 'classic'
local m = classic.module(...)

local py = require('fb.python')

py.exec([=[
import scipy.stats as stats
import numpy as np

def truncnorm(shape,a,b,mu,sigma):
    T = stats.truncnorm(
        (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    return T.rvs(shape)
]=])

function m.truncnorm(shape,a,b,mu,sigma)
  return py.eval('truncnorm(shape,a,b,mu,sigma)',{shape = shape, a=a,b=b,mu=mu,sigma=sigma})
end

return m
