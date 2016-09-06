local classic = require 'classic'

local m = classic.module(...)

m:class('simulator')

function m.create_views(tensor,positions,view_size,stack_dimensions)
  -- pprint(tensor)
  -- pprint(positions)
  -- pprint(view_size)
  --
  -- print('stackdim ' .. stack_dimensions)
  local stack_dim = stack_dimensions or 1
  local views = {}
  for i=1,positions:size(1) do
    local d = 1
    local slice = {}
    for ddim = 1,stack_dim do
      slice[d] = {}
      d = d + 1
    end
    slice[d] = {positions[i][1],positions[i][1]+view_size-1}
    d = d + 1
    slice[d] = {positions[i][2],positions[i][2]+view_size-1}
    -- pprint(slice)
    views[i] = tensor[slice]
  end
  return views
end

return m
