local classic = require 'classic'

local tabular_schedule = classic.class(...)
function tabular_schedule:_init(schedule)
  self.schedule = schedule
end

function tabular_schedule:__call(it)
  for k, v in ipairs(self.schedule) do
    local start = v[1]
    local ends = v[2]
    local retval = v[3]
    if it >= start and it < ends then
      return retval
    end
  end
end
