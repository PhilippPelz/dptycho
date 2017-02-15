local classic = require 'classic'

local reg_schedule = classic.class(...)
function reg_schedule:_init(start,iterations,startval,stopval)
  self.start = start
  self.iterations = iterations
  self.startval = startval
  self.stopval = stopval
end

function reg_schedule:__call(it)
  if it < self.start or it > self.start + self.iterations then return false
  else
    return self.startval - (self.startval - self.stopval) * (it-self.start)/self.iterations
  end
end
