----------------------------------------------------------------------
--
-- Deep Genetic Programming: Reifying an AI researcher.
--
-- Functions for data import
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'image'
require 'torch'

----------------------------------------------------------------------
-- Basic import function for SVHN with normalization
function import_data()
  channels = {'y','u','v'};
  -- Sets names
  setFiles = { train='svhn/train_32x32.t7', test='svhn/test_32x32.t7'};
  sets = {};
  -- Load the datasets (factored)
  for key,value in pairs(setFiles) do
      print("    - Loading " .. value);
      -- Load the matalb version
      tmp = torch.load(value, 'ascii');
      -- Transform to row-major
      curData = {
          data = tmp.X:transpose(3,4):float(),
          labels = tmp.y[1],
          mean = {},
          std = {},
          size = function () return (data:size(1)) end
      };
      collectgarbage();
      print("        . Transform to YUV");
      -- Pre-processing data to YUV
      for i = 1,curData.data:size(1) do
          curData.data[i] = image.rgb2yuv(curData.data[i]);
      end
      print("        . Channel-wise normalization");
      -- Channel-wise normalization
      for i,name in ipairs(channels) do
          curData.mean[i] = curData.data[{{}, i, {}, {}}]:mean();
          curData.std[i] = curData.data[{{}, i, {}, {}}]:std();
          curData.data[{{}, i, {}, {}}] = (curData.data[{{}, i, {}, {}}] - curData.mean[i]) / curData.std[i];
      end
      sets[key] = curData;
      collectgarbage();
  end
  ----------------------------------------------------------------------
  -- Data selection (if reduced data option)
  trShuffle = torch.randperm(sets["train"].data:size(1));
  sets["train"].data = sets["train"].data[{{1,trsize},{},{},{}}];
  trShuffle = torch.randperm(sets["test"].data:size(1));
  sets["test"].data = sets["test"].data[{{1,tesize},{},{},{}}];
  ----------------------------------------------------------------------
  -- Data normalization (on Y channel)
  print "    - Contrastive normalization"
  neighborhood = image.gaussian1D(7);
  -- Define the normalization operator (can be inserted inside training model)
  normalization = nn.SpatialContrastiveNormalization(1,neighborhood):float();
  -- Apply this gaussian normalization
  for key,value in pairs(sets) do
      for i = 1, sets[key].data:size(1) do
          sets[key].data[{i, {1}, {}, {}}] = normalization:forward(sets[key].data[{i, {1}, {}, {}}]);
      end
  end
  print "    - Checking data statistics";
  for key,value in pairs(sets) do
    for i,channel in ipairs(channels) do
      meanData = sets[key].data[{ {},i }]:mean();
      stdData = sets[key].data[{ {},i }]:std();
      print('    - '..key..' data, '..channel..'-channel, mean: ' .. meanData .. ', standard deviation: ' .. stdData);
    end
  end
end