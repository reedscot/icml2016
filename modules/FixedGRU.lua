
local FixedGRU = {}
function FixedGRU.rnn(nstep, avg, emb_dim)
  if avg == nil then
    avg = 0
  end
  if emb_dim == nil then
    emb_dim = 256
  end
  local inputs = {}
  for n = 1,nstep do
    table.insert(inputs, nn.Identity()())
  end

  -- gates for update
  local i2h_update = {}
  local h2h_update = {}
  -- gates for reset
  local i2h_reset = {}
  local h2h_reset = {}
  -- candidate hidden state
  local i2h = {}
  local h2h = {}
  -- actual hidden state
  local hids = {}

  for i,v in ipairs(inputs) do
    i2h_update[i] = nn.Sequential()
    i2h_update[i]:add(nn.Linear(emb_dim,emb_dim))
    i2h_reset[i] = nn.Sequential()
    i2h_reset[i]:add(nn.Linear(emb_dim,emb_dim))
    i2h[i] = nn.Sequential()
    i2h[i]:add(nn.Linear(emb_dim,emb_dim))

    if i > 1 then
      i2h_update[i]:share(i2h_update[1],'weight', 'bias', 'gradWeight', 'gradBias')
      i2h_reset[i]:share(i2h_reset[1],'weight', 'bias', 'gradWeight', 'gradBias')
      i2h[i]:share(i2h[1], 'weight', 'bias', 'gradWeight', 'gradBias')

      h2h_update[i-1] = nn.Sequential()
      h2h_update[i-1]:add(nn.Linear(emb_dim,emb_dim))
      h2h_reset[i-1] = nn.Sequential()
      h2h_reset[i-1]:add(nn.Linear(emb_dim,emb_dim))
      h2h[i-1] = nn.Sequential()
      h2h[i-1]:add(nn.Linear(emb_dim,emb_dim))

      if i > 2 then
        h2h_update[i-1]:share(h2h_update[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
        h2h_reset[i-1]:share(h2h_reset[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
        h2h[i-1]:share(h2h[i-2],'weight', 'bias', 'gradWeight', 'gradBias')
      end

      -- compute update and reset gates.
      update = nn.Sigmoid()(nn.CAddTable()({i2h_update[i](inputs[i]), h2h_update[i-1](hids[i-1])}))
      reset = nn.Sigmoid()(nn.CAddTable()({i2h_reset[i](inputs[i]), h2h_reset[i-1](hids[i-1])}))

      -- compute candidate hidden state.
      local gated_hidden = nn.CMulTable()({reset, hids[i-1]})
      local p1 = i2h[i](inputs[i])
      local p2 = h2h[i-1](gated_hidden)
      local hidden_cand = nn.Tanh()(nn.CAddTable()({p1, p2}))

      -- use gates to interpolate hidden state.
      local zh = nn.CMulTable()({update, hidden_cand})
      local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update)), hids[i-1]})
      hids[i] = nn.CAddTable()({zh, zhm1})
    else
      hids[i] = nn.Tanh()(i2h[i](inputs[i]))
    end
  end 

  local hid
  if avg == 1 then
    hid = hids[1]
    for n = 2,nstep do
      hid = nn.CAddTable()({hid, hids[n]})
    end
    hid = nn.MulConstant(1./nstep)(hid)
  else
    hid = hids[#hids]
  end

  local outputs = {}
  table.insert(outputs, hid)
  return nn.gModule(inputs, outputs)
end

return FixedGRU

