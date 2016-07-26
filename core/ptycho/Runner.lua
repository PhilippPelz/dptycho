local classic = require 'classic'
local Runner = classic.class(...)

-- takes config table
-- {{iterations, engine1},
--  {iterations, engine2}}
function Runner:_init(config,params)
  self.config = config
  self.params = params
  return self
end

function Runner:run()
  for _, entry in ipairs(self.config) do
    local iterations = entry[1]
    local engine = entry[2]
    if entry[3] then
      local param_modifier = entry[3]
      param_modifier(self.params)
    end
    local engine_instance = engine(self.params)
    engine_instance:iterate(iterations)
    engine_instance = nil
    collectgarbage()
  end
end

return Runner
