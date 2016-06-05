
local HybridCNNLong = {}
function HybridCNNLong.cnn(alphasize, emb_dim, dropout, avg, cnn_dim)
  dropout = dropout or 0.0
  avg = avg or 0
  cnn_dim = cnn_dim or 256

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Bag

  local net = nn.Sequential()
  -- 201 x alphasize
  net:add(nn.TemporalConvolution(alphasize, 384, 4))
  net:add(nn.Threshold())
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 66 x 256
  net:add(nn.TemporalConvolution(384, 512, 4))
  net:add(nn.Threshold())
  net:add(nn.TemporalMaxPooling(3, 3))
  -- 21 x 256
  net:add(nn.TemporalConvolution(512, cnn_dim, 4))
  net:add(nn.Threshold())
  local h1 = nn.SplitTable(2)(net(inputs[1]))

  local r2 = FixedRNN.rnn(18, avg, cnn_dim)(h1)
  out = nn.Linear(cnn_dim, emb_dim)(nn.Dropout(dropout)(r2))
  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return HybridCNNLong

