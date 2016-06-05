--[[ 

Train a text encoder.

--]]
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

opt = {
  numCaption = 1,
  encoder = 'rnn',
  cnn_dim = 512,
  save_every = 100,
  print_every = 1,
  dataset = 'coco',
  img_dir = '',
  filenames = '',
  data_root = '/home/reedscot/data/cub_files6',
  checkpoint_dir = '/home/reedscot/checkpoints',
  batchSize = 64,
  doc_length = 201,
  loadSize = 76,
  txtSize = 1024,         -- #  of dim for raw text.
  fineSize = 64,
  nThreads = 4,           -- #  of data loading threads to use
  niter = 1000,             -- #  of iter at starting learning rate
  lr = 0.0002,            -- initial learning rate for adam
  lr_decay = 0.5,            -- initial learning rate for adam
  decay_every = 100,
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
  display = 1,            -- display samples while training. 0 = false
  display_id = 10,        -- display window id.
  gpu = 2,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  name = 'coco',
  init_t = '',
  use_cudnn = 0,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if opt.gpu > 0 then
  ok, cunn = pcall(require, 'cunn')
  ok2, cutorch = pcall(require, 'cutorch')
  cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

FixedRNN = require('modules.FixedGRU')
DocumentCNN = require('modules.HybridCNNLong')

if opt.init_t == '' then
  alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
  dict = {}
  for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
  end
  alphabet_size = #alphabet
  netT = DocumentCNN.cnn(alphabet_size, opt.txtSize, 0, 1, opt.cnn_dim)
  netT:apply(weights_init)
else
  netT = torch.load(opt.init_t)
end

---------------------------------------------------------------------------
optimStateT = {
  learningRate = opt.lr,
  beta1 = opt.beta1,
}
----------------------------------------------------------------------------

local input_txt_raw1 = torch.zeros(opt.batchSize, opt.doc_length, alphabet_size)
local input_txt_raw2 = torch.zeros(opt.batchSize, opt.doc_length, alphabet_size)

local input_txt_emb1 = torch.Tensor(opt.batchSize, opt.txtSize)
local input_txt_emb2 = torch.Tensor(opt.batchSize, opt.txtSize)

local errT

local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
  input_txt_emb1 = input_txt_emb1:cuda()
  input_txt_emb2 = input_txt_emb2:cuda()
  input_txt_raw1 = input_txt_raw1:cuda()
  input_txt_raw2 = input_txt_raw2:cuda()
  netT:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netT = cudnn.convert(netT, cudnn)
end

local parametersT, gradParametersT = netT:getParameters()

if opt.display then disp = require 'display' end

local sample = function()
  data_tm:reset(); data_tm:resume()
  real_img, wrong_img, real_txt, match_txt = data:getBatch()
  data_tm:stop()

  input_txt_raw1:copy(real_txt)
  input_txt_raw2:copy(match_txt)
end

acc_batch = 0.0
acc_smooth = 0.0
function JointEmbeddingLoss(fea_txt1, fea_txt2)
  local batch_size = fea_txt1:size(1)
  local score = torch.zeros(batch_size, batch_size)
  local txt1_grads = fea_txt1:clone():fill(0)
  local txt2_grads = fea_txt2:clone():fill(0)

  local loss = 0
  acc_batch = 0.0
  for i = 1,batch_size do
    for j = 1,batch_size do
      score[{i,j}] = torch.dot(fea_txt2:narrow(1,i,1), fea_txt1:narrow(1,j,1))
    end
    local label_score = score[{i,i}]
    for j = 1,batch_size do
      if (i ~= j) then
        local cur_score = score[{i,j}]
        local thresh = cur_score - label_score + 1
        if (thresh > 0) then
          loss = loss + thresh
          local txt_diff = fea_txt1:narrow(1,j,1) - fea_txt1:narrow(1,i,1)
          txt2_grads:narrow(1, i, 1):add(txt_diff)
          txt1_grads:narrow(1, j, 1):add(fea_txt2:narrow(1,i,1))
          txt1_grads:narrow(1, i, 1):add(-fea_txt2:narrow(1,i,1))
        end
      end 
    end
    local max_score, max_ix = score:narrow(1,i,1):max(2)
    if (max_ix[{1,1}] == i) then
      acc_batch = acc_batch + 1
    end
  end
  acc_batch = 100 * (acc_batch / batch_size)
  local denom = batch_size * batch_size
  local res = { [1] = txt1_grads:div(denom),
                [2] = txt2_grads:div(denom) }
  acc_smooth = 0.99 * acc_smooth + 0.01 * acc_batch
  return loss / denom, res
end

local fTx = function(x)
  gradParametersT:zero()

  -- real txt
  input_txt_emb1:copy(netT:forward(input_txt_raw1))
  -- get matching text embeddings
  input_txt_emb2:copy(netT:forward(input_txt_raw2))

  errT, grads = JointEmbeddingLoss(input_txt_emb1, input_txt_emb2)

  netT:backward(input_txt_raw2, grads[2])

  netT:forward(input_txt_raw1)
  netT:backward(input_txt_raw1, grads[1])

  return errT, gradParametersT
end

-- train
for epoch = 1, opt.niter do
  epoch_tm:reset()

  if epoch % opt.decay_every == 0 then
    optimStateT.learningRate = optimStateT.learningRate * opt.lr_decay
  end

  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    tm:reset()

    sample()
    optim.adam(fTx, parametersT, optimStateT)

    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. ' T:%.3f, acc:(%.3f,%.3f)'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateT.learningRate,
              errT and errT or -1,
              acc_batch, acc_smooth))
    end
  end
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_T.t7', netT)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_opt.t7', opt)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end

