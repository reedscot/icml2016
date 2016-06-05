local util = {}
require 'cunn'
local ffi=require 'ffi'

-- returns bbox in normalized coordinates.
function util.affine_to_bbox(imgdim, affine)
  local bbox = torch.zeros(4, affine:size(1))
  for b = 1,bbox:size(2) do
    local scale_Y = affine[{b,1,1}]
    local scale_X = affine[{b,2,2}]
    local t_y = affine[{b,1,3}]
    local t_x = affine[{b,2,3}]

    bbox[{1,b}] = 0.5 + t_x / 2.0 - scale_X / 2.0
    bbox[{2,b}] = 0.5 + t_y / 2.0 - scale_Y / 2.0
    bbox[{3,b}] = scale_X
    bbox[{4,b}] = scale_Y

    --print(string.format('scale_Y: %.2f scale_X: %.2f t_y: %.2f t_x: %.2f',
    --      scale_Y, scale_X, t_y, t_x))
    --print(bbox[{{},b}])
    --debug.debug()
  end
  return bbox
end

-- assumes bbox is in normalized coordinates.
function util.bbox_to_affine(imgdim, bbox)
  local affine = torch.zeros(bbox:size(2), 2, 3)
  for b = 1,bbox:size(2) do
    local b_x = bbox[{1,b}]
    local b_y = bbox[{2,b}]
    local bW =  loc[{3,b}]
    local bH =  loc[{4,b}]
    local t_x = 2*((b_x + bW/2.0) - 0.5)
    local t_y = 2*((b_y + bH/2.0) - 0.5)
    local scale_X = bW
    local scale_Y = bH
    loc_affine[{b,1,1}] = scale_Y
    loc_affine[{b,2,2}] = scale_X
    loc_affine[{b,1,3}] = t_y
    loc_affine[{b,2,3}] = t_x
  end
  return affine
end

function util.invert_affine(affine)
  local inv_aff = affine:clone()
  for b = 1,affine:size(1) do
    inv_aff[{b,1,1}] = 1.0/affine[{b,1,1}]
    inv_aff[{b,2,2}] = 1.0/affine[{b,2,2}]
    inv_aff[{b,1,3}] = -affine[{b,1,3}] / affine[{b,1,1}]
    inv_aff[{b,2,3}] = -affine[{b,2,3}] / affine[{b,2,2}]
  end
  return inv_aff
end

-- assumes bbox is in normalized coordinates.
function util.draw_box(img, bbox, thickness)
  thickness = thickness or 3
  local boxcolor = torch.zeros(3)
  boxcolor[3] = 1

  local iW = img:size(3)
  local iH = img:size(2)

  local x = math.floor(bbox[1] * iW)
  local y = math.floor(bbox[2] * iH)
  local width = math.floor(bbox[3] * iW)
  local height = math.floor(bbox[4] * iH)

  -- top bar
  for n = 1,width do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((y + offset) <= img:size(2) and  (y + offset) >= 1) then
        if (x+n-1 <= img:size(3) and (x+n-1 > 0)) then
          img[{{},y + offset, x+n-1}]:copy(boxcolor)
        end
      end
    end
  end
  -- bottom bar
  for n = 1,width do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((y + height + offset) <= img:size(2) and (y + height + offset) >= 1) then
        if (x+n-1 <= img:size(3) and (x+n-1 > 0)) then
          img[{{},y + height + offset, x+n-1}]:copy(boxcolor)
        end
      end
    end
  end
  -- left bar
  for n = 1,height do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((x + offset) <= img:size(3) and (x + offset) >= 1) then
        if (y+n-1 <= img:size(2) and (y+n-1 > 0)) then
          img[{{},y+n-1, x + offset}]:copy(boxcolor)
        end
      end
    end
  end
  -- right bar
  for n = 1,height do
    for t = 1,thickness do
      local offset = -math.floor(thickness / 2) + (t-1)
      if ((x + width + offset) <= img:size(3) and (x + width + offset) >= 1) then
        if (y+n-1 <= img:size(2) and (y+n-1 > 0)) then
          img[{{},y+n-1, x + width + offset}]:copy(boxcolor)
        end
      end
    end
  end

  return img
end

function util.save(filename, net, gpu)

    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end

    for k, l in ipairs(netsave.modules) do
        -- convert to CPU compatible model
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
					      l.kW, l.kH, l.dW, l.dH, 
					      l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
					       l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end

        -- clean up buffers
        local m = netsave.modules[k]
        m.output = m.output.new()
        m.gradInput = m.gradInput.new()
        m.finput = m.finput and m.finput.new() or nil
        m.fgradInput = m.fgradInput and m.fgradInput.new() or nil
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
	-- TODO: figure out why giant storage-offsets being created on typecast
        if m.weight then 
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
    netsave.output = netsave.output.new()
    netsave.gradInput = netsave.gradInput.new()

    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    torch.save(filename, netsave)
end

function util.load(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
   return net
end

function util.cudnn(net)
    for k, l in ipairs(net.modules) do
        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
						 l.kW, l.kH, l.dW, l.dH, 
						 l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
    return net
end

-- a function to do memory optimizations by 
-- setting up double-buffering across the network.
-- this drastically reduces the memory needed to generate samples.
function util.optimizeInferenceMemory(net)
    local finput, output, outputB
    net:apply(
        function(m)
            if torch.type(m):find('Convolution') then
                finput = finput or m.finput
                m.finput = finput
                output = output or m.output
                m.output = output
            elseif torch.type(m):find('ReLU') then
                m.inplace = true
            elseif torch.type(m):find('BatchNormalization') then
                outputB = outputB or m.output
                m.output = outputB
            end
    end)
end

function util.makeDataParallel(model, nGPU)
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)

      for i=1, nGPU do
         cutorch.setDevice(opt.gpu + i - 1)
         model:add(model_single:clone():cuda(), opt.gpu + i - 1)
      end
   end
   cutorch.setDevice(opt.gpu)
   return model
end

function util.cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opt.gpu)
   newDPT:add(module:get(1), opt.gpu)
   return newDPT
end

function util.saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, util.cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(util.cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function util.loadDataParallel(filename, nGPU)
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return util.makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = util.makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

return util
