require 'image'
dir = require 'pl.dir'

trainLoader = {}

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end
alphabet_size = #alphabet

function decode(txt)
  local str = ''
  for w_ix = 1,txt:size(1) do
    local ch_ix = txt[w_ix]
    local ch = ivocab[ch_ix]
    if (ch  ~= nil) then
      str = str .. ch
    end
  end
  return str
end

function trainLoader:decode2(txt)
  local str = ''
  _, ch_ixs = txt:max(2)
  for w_ix = 1,txt:size(1) do
    local ch_ix = ch_ixs[{w_ix,1}]
    local ch = ivocab[ch_ix]
    if (ch ~= nil) then
      str = str .. ch
    end
  end
  return str
end

trainLoader.alphabet = alphabet
trainLoader.alphabet_size = alphabet_size
trainLoader.dict = dict
trainLoader.ivocab = ivocab
trainLoader.decode = decoder

local files = {}
local size = 0
cur_files = dir.getfiles(opt.data_root)
size = size + #cur_files

--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, loadSize[2], loadSize[2])
  return input
end

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(path)
  collectgarbage()
  local input = loadImage(path)
  if opt.no_aug == 1 then
    return image.scale(input, sampleSize[2], sampleSize[2])
  end

  local iW = input:size(3)
  local iH = input:size(2)

  -- do random crop
  local oW = sampleSize[2];
  local oH = sampleSize[2]
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
  assert(out:size(2) == oW)
  assert(out:size(3) == oH)
  -- do hflip with probability 0.5
  if torch.uniform() > 0.5 then out = image.hflip(out); end
  -- rotation
  if opt.rot_aug then
    local randRot = torch.rand(1)*0.16-0.08
    out = image.rotate(out, randRot:float()[1], 'bilinear')
  end
  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
  return out
end

function trainLoader:sample(quantity)
  local ix_file1 = torch.Tensor(quantity)
  local ix_file2 = torch.Tensor(quantity)
  for n = 1, quantity do
    local samples = torch.randperm(#cur_files):narrow(1,1,2)
    local file_ix1 = samples[1]
    local file_ix2 = samples[2]
    ix_file1[n] = file_ix1
    ix_file2[n] = file_ix2
  end

  local data_img1 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- real
  local data_img2 = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2]) -- mismatch
  local data_txt1 = torch.zeros(quantity, opt.doc_length, alphabet_size) -- real
  local data_txt2 = torch.zeros(quantity, opt.doc_length, alphabet_size) -- match

  for n = 1, quantity do
    local t7file1 = cur_files[ix_file1[n]]
    local info1 = torch.load(t7file1)
    local t7file2 = cur_files[ix_file2[n]]
    local info2 = torch.load(t7file2)

    local img_file1 = opt.img_dir .. '/' .. info1.img
    local img1 = trainHook(img_file1)
    local img_file2 = opt.img_dir .. '/' .. info2.img
    local img2 = trainHook(img_file2)
 
    local txt_sample = torch.randperm(info1.char:size(2))
    local ix_txt1 = txt_sample[1]
    local ix_txt2 = txt_sample[2]
    for t = 1,opt.doc_length do
      -- real text
      local on_ix = info1.char[{t,ix_txt1}]
      if on_ix ~= 0 then
        data_txt1[{n,t,on_ix}] = 1
      end
      -- matching text
      local on_ix = info1.char[{t,ix_txt2}]
      if on_ix ~= 0 then
        data_txt2[{n,t,on_ix}] = 1
      end
    end
    -- real image
    data_img1[n]:copy(img1)
    -- mis-match image
    data_img2[n]:copy(img2)
  end
  collectgarbage(); collectgarbage()
  return data_img1, data_img2, data_txt1, data_txt2
end

function trainLoader:size()
  return size
end

