----------------------------------------------------------------------
--
-- Deep Genetic Programming: Reifying an AI researcher.
--
-- Set of modules the researcher has access to and init functions
--    * Initialize tables
--    * Setup list of models
--    * Processing functions
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Sets of libs requirements

require 'optim'
require 'torch'
require 'dp'
--require 'cunn'
require 'rnn'
--require 'fbnn'
require 'nngraph'
require 'graph'
require 'nnx'
require 'nn'

----------------------------------------------------------------------
-- classes parameters
classes = {'1','2','3','4','5','6','7','8','9','0'}
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

----------------------------------------------------------------------
-- Add the potential modules to the table of available modules
function insert_module(moduleName, moduleCall, moduleType, nParameters, paramNames, paramRange)
	-- ensure we haven't already inserted this function
	if module_table[moduleName] ~= nil then
		print("Warning - module already defined: " .. moduleName)
		return false
	end
	-- Add this function to the table
	tempFunction = {}
	tempFunction.func = moduleCall
	tempFunction.name = moduleName
	tempFunction.type = moduleType
	tempFunction.params = nParameters
  tempFunction.paramRange = paramRange
  tempFunction.paramNames = paramNames
	table.insert(module_table, tempFunction);
	return true
end

----------------------------------------------------------------------
-- Sample the input to a certain percent
function sampleInput(input, percent)
  setSize = input.data:size(1);
  shuffle = torch.randperm(setSize)
  return input.data[{shuffle[{1,(percent*setSize)}],{},{},{}}];
end

----------------------------------------------------------------------
-- List of parameter ranges
minUnits = 10;
maxUnits = 2000;

----------------------------------------------------------------------
-- List of available functions
function establish_functions()
  -- Basic input/output (identity) nodes
  insert_module("input", nn.Identity, "input", 0, {}, {});
  insert_module("output", nn.Identity, "output", 0, {}, {});
  ---------------------
  -- Tensors containers
  ---------------------
  -- Sequential densely connects modules in a feed-forward manner
  insert_module("Sequential", nn.Sequential, "iterate", 0, {}, {});
  -- Parallel applies its ith child module to the ith slice of the input Tensor by using select on dimension inputDimension
  insert_module("Parallel", nn.Parallel, "iterate", 2, {'dimensionIn', 'dimensionOut'}, {{1, ndims}, {1, ndims}});
  -- Concat concatenates the output of one layer of "parallel" modules along the provided dimension dim
  insert_module("Concat", nn.Concat, "iterate", 1, {'dimensionIn'}, {{1, ndims}});
  -- DepthConcat concatenates output of parallel layer through depth (can be different dimensionalities)
  insert_module("DepthConcat", nn.DepthConcat, "iterate", 1, {'dimensionIn'}, {{1, ndims}});
  ---------------------
  -- Transfer functions
  ---------------------
  -- Hard hyperbolic tangent
	insert_module("HardTanh", nn.HardTanh, "transfer", 0, {}, {});
  -- Hard shrinkage (almost RELU with threshold-based)
	insert_module("HardShrink", nn.HardShrink, "transfer", 1, {'lambda'}, {{-1.5,1.5});
  -- Soft shrinkage (threshold-based)
	insert_module("SoftShrink", nn.SoftShrink, "transfer", 1, {'lambda'}, {{-1.5,1.5});
  -- Softmax function (and unit-sum rescaling)
	insert_module("SoftMax", nn.SoftMax, "transfer", 0, {}, {});
  -- Softmin function (and unit-sum rescaling)
	insert_module("SoftMin", nn.SoftMin, "transfer", 0, {}, {});
  -- Softplus function (and unit-sum rescaling)
	insert_module("SoftPlus", nn.SoftPlus, "transfer", 0, {}, {});
  -- Softsign function always positive
	insert_module("SoftSign", nn.SoftSign, "transfer", 0, {}, {});
  -- Log-Sigmoid transfer function
	insert_module("LogSigmoid", nn.LogSigmoid, "transfer", 0, {}, {});
  -- Log-Softmax transfer function
	insert_module("LogSoftMax", nn.LogSoftMax, "transfer", 0, {}, {});
  -- Sigmoid transfer function
	insert_module("Sigmoid", nn.Sigmoid, "transfer", 0, {}, {});
  -- Tanh transfer function
	insert_module("Tanh", nn.Tanh, "transfer", 0, {}, {});
  -- Rectified Linear Units transfer function
	insert_module("ReLU", nn.ReLU, "transfer", 0, {}, {});
  -- Parametric Rectified Linear Units transfer function
	insert_module("PReLU", nn.PReLU, "transfer", 0, {}, {});
  ---------------------
  -- Combination functions
  ---------------------
  -- Linear combination of the inputs
  insert_module("Linear", nn.Linear, "transform", 2, {'inputDimension', 'outputDimension'}, {{minUnits, maxUnits}, {minUnits, maxUnits}});
  -- Sparse linear combination of the inputs
  insert_module("SparseLinear", nn.SparseLinear, "transform", 2, {'inputDimension', 'outputDimension'}, {{minUnits, maxUnits}, {minUnits, maxUnits}});
  -- Dropout module
	insert_module("Dropout", nn.Dropout, "transform", 1, {'ratio'}, {{0, 1}});
  -- Spatial Dropout module
	insert_module("SpatialDropout", nn.SpatialDropout, "transform", 1, {'ratio'}, {{0, 1}});
  ---------------------
  -- Mathematical functions
  ---------------------
  -- Absolute value of the input
	insert_module("Abs", nn.Abs, "mathematical", 0, {}, {});
  -- Add a scalar to the input
	insert_module("Add", nn.Add, "mathematical", 2, {'inputDimension', 'scalar'}, {{minUnits, maxUnits}, {minUnits, maxUnits}});
  -- Multiply the input
	insert_module("Mul", nn.Mul, "mathematical", 0, {}, {});
  -- Component-wise multiply the input
	insert_module("CMul", nn.CMul, "mathematical", 1, {'size'}, {{1, ndims}});
  -- Min across a specific dimensions of the input
	insert_module("Min", nn.Min, "mathematical", 1, {'size'}, {{1, ndims}});
  -- Min across a specific dimensions of the input
	insert_module("Max", nn.Max, "mathematical", 1, {'size'}, {{1, ndims}});
  -- Min across a specific dimensions of the input
	insert_module("Mean", nn.Mean, "mathematical", 1, {'size'}, {{1, ndims}});
  -- Min across a specific dimensions of the input
	insert_module("Sum", nn.Sum, "mathematical", 1, {'size'}, {{1, ndims}});
  -- Euclidean distance of the input to outputSize centers
  insert_module("Euclidean", nn.Euclidean, "mathematical", 2, {'inputDimension', 'outputDimension'}, {{minUnits, maxUnits}, {minUnits, maxUnits}});
  -- Euclidean distance which additionally learns a separate diagonal covariance matrix
  insert_module("WeightedEuclidean", nn.WeightedEuclidean, "mathematical", 2, {'inputDimension', 'outputDimension'}, {{minUnits, maxUnits}, {minUnits, maxUnits}});
  -- Exponentiate the input
	insert_module("Exp", nn.Exp, "mathematical", 0, {}, {});
  -- Square power of the input
	insert_module("Square", nn.Square, "mathematical", 0, {}, {});
  -- Square root of the input
	insert_module("Sqrt", nn.Sqrt, "mathematical", 0, {}, {});
  -- Take the power of the input
	insert_module("Power", nn.Power, "mathematical", 1, {'power'}, {{1,32}});
  -- Batch normalization of the input
	insert_module("BatchNormalization", nn.BatchNormalization, "mathematical", 0, {}, {});
      -- Take the power of the input
	insert_module("L1Penalty", nn.BatchNormalization, "mathematical", 1, {'L1weight', 'sizeAverage'}, {{1,32}});
  ---------------------
  -- Dimensional functions
  ---------------------
  -- Narrow the module to a given length starting at an offset
  insert_module("Narrow", nn.Narrow, "dimensional", 3, {'inputDimension', 'offset', 'outputDimension'}, {{minUnits, maxUnits}, {minUnits, maxUnits}, {minUnits, maxUnits}});
  -- Replicate the module to a given length starting at an offset
  insert_module("Replicate", nn.Replicate, "dimensional", 2, {'replication', 'dimension'}, {{2, maxUnits}, {1, ndims}});
  ---------------------
  -- Table functions
  ---------------------
  -- ConcatTable()            : applies each member module to the same input Tensor and outputs a table;
  -- ParallelTable()          : applies the i-th member module to the i-th input and outputs a table;
  -- SplitTable(dim,nInputs)  : splits a Tensor into a table of Tensors;
  -- JoinTable(dim,nInputs)   : joins a table of Tensors into a Tensor;
  -- MixtureTable(dim)        : mixture of experts weighted by a gater;
  -- SelectTable(index)       : select one element from a table;
  -- NarrowTable(offset,len)  : select a slice of elements from a table;
  -- FlattenTable()           : flattens a nested table hierarchy;
  -- PairwiseDistance(p)      : outputs the p-norm distance between inputs;
  -- DotProduct()             : outputs the dot product (similarity) between inputs;
  -- CosineDistance()         : outputs the cosine distance between inputs;
  -- CAddTable()              : addition of input Tensors;
  -- CSubTable()              : substraction of input Tensors;
  -- CMulTable()              : multiplication of input Tensors;
  -- CDivTable()              : division of input Tensors
  ---------------------
  -- Temporal convolutions (1-dimensional sequences)
  ---------------------
  -- TemporalConvolution (1D convolution over an input sequence)
  insert_module("TemporalConvolution", nn.TemporalConvolution, "temporal-convolution", 4,  {'inputFrameSize', 'outputFrameSize', 'kernelWidth', 'convStep'}, {{1, 64}, {1, 64}, {1, 64}, {1, 64}});
  -- TemporalSubSampling (1D sub-sampling over an input sequence)
  insert_module("TemporalSubSampling", nn.TemporalSubSampling, "temporal-convolution", 3,  {'inputFrameSize', 'kernelWidth', 'convStep'}, {{1, 64}, {1, 64}, {1, 64}});
  -- TemporalMaxPooling (1D max-pooling operation over an input sequence)
  insert_module("TemporalMaxPooling", nn.TemporalMaxPooling, "temporal-convolution", 2,  {'kernelWidth', 'convStep'}, {{1, 64}, {1, 64}});
  -- LookupTable (convolution of width 1 usually for word embeddings)
  insert_module("LookupTable", nn.LookupTable, "temporal-convolution", 2,  {'nIndex', 'sizes'}, {{1, ninputs}, {1, ndims}});
  ---------------------
  -- Spatial convolutions (2-dimensional sequences)
  ---------------------
  -- SpatialConvolution (2D convolution over an input image)
  insert_module("SpatialConvolution", nn.SpatialConvolution, "spatial-convolution", 4,  {'inputPlane', 'outputPlane', 'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}, {1, 64}, {1, 64}});
  -- SpatialSubSampling (2D sub-sampling over an input image)
  insert_module("SpatialSubSampling", nn.SpatialSubSampling, "spatial-convolution", 3,  {'inputPlane', 'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}, {1, 64}});
  -- SpatialMaxPooling (2D max-pooling operation over an input image)
  insert_module("SpatialMaxPooling", nn.SpatialMaxPooling, "spatial-convolution", 2,  {'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}});
  -- SpatialAveragePooling (2D average-pooling operation over an input image)
  insert_module("SpatialAveragePooling", nn.SpatialAveragePooling, "spatial-convolution", 2,  {'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}});
  -- SpatialAdaptiveMaxPooling (2D max-pooling operation which adapts its parameters dynamically)
  insert_module("SpatialAdaptiveMaxPooling", nn.SpatialAdaptiveMaxPooling, "spatial-convolution", 2,  {'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}});
  -- SpatialLPPooling (p-norm in a convolutional manner on a set of input images)
  insert_module("SpatialLPPooling", nn.SpatialLPPooling, "spatial-convolution", 4,  {'inputPlane', 'pNorm', 'kernelWidth', 'convStep'}, {{1, 64}, {1, 64}, {1, 64}, {1, 64}});
  -- SpatialZeroPadding (padds a feature map with specified number of zeros)
  insert_module("SpatialZeroPadding", nn.SpatialZeroPadding, "spatial-convolution", 4,  {'padLeft', 'padRight', 'padTop', 'padBottom'}, {{1, 64}, {1, 64}, {1, 64}, {1, 64}});
  -- SpatialSubtractiveNormalization (spatial subtraction operation on a series of 2D inputs)
  insert_module("SpatialSubtractiveNormalization", nn.SpatialSubtractiveNormalization, "spatial-convolution", 2,  {'ninputplane', 'kernel'}, {{1, 64}, {1, 64}});
  -- SpatialBatchNormalization (mean/std normalization over a mini-batch inputs)
	insert_module("SpatialBatchNormalization", nn.SpatialBatchNormalization, "spatial-convolution", 0, {}, {});
  ---------------------
  -- Volumetric convolutions (3-dimensional sequences)
  ---------------------
  -- VolumetricConvolution (3D convolution over video)
  insert_module("VolumetricConvolution", nn.VolumetricConvolution, "volumetric-convolution", 5,  {'inputPlane', 'outputPlane', 'kernelTime', 'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}, {1, 64}, {1, 64}, {1, 64}});
  -- VolumetricMaxPooling (3D max-pooling over video).
  insert_module("VolumetricMaxPooling", nn.VolumetricMaxPooling, "volumetric-convolution", 3,  {'kernelTime', 'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}, {1, 64}});
  -- VolumetricAveragePooling (3D average-pooling over video)
  insert_module("VolumetricAveragePooling", nn.VolumetricAveragePooling, "volumetric-convolution", 3,  {'kernelTime', 'kernelWidth', 'kernelHeight'}, {{1, 64}, {1, 64}});
  ---------------------
  -- Multiple classifications (from nnx)
  ---------------------
  -- SoftMaxTree (A hierarchy of parameterized log-softmaxes, useful for very wide number of classes, use wutg TreeLLCCriterion)
  insert_module("SoftMaxTree", nn.SoftMaxTree, "multi-classification", 3,  {'inputSize', 'hierarchy'}, {{1, 64}, {1, 64}});
  -- MultiSoftMax performs a softmax over the last dimension of 2/3-dimensional tensor
  ---------------------
  -- Recurrent modules
  ---------------------
  --
  --
  --
end

----------------------------------------------------------------------
-- Process the current network
function evaluateNetwork(model, trainData, testData)
  -- First compute the forward activation
  inTrans = model:forward(trainData.data)
  print(inTrans)
  -- Then add a softmax on this transform
  softModel = nn.Sequential();
  softModel:add(nn.LogSoftMax());
  criterion = nn.ClassNLLCriterion();
  -- shuffle at each epoch
  shuffle = torch.randperm(trsize);
  -- set model to train mode
  --softModel:train()
  for e = 1,opt.epochs do
    for t = 1,inTrans:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, inTrans:size())
      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = inTrans[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
        -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end
        -- reset gradients
        gradParameters:zero()
        -- f is the average of all criterions
        local f = 0
        -- evaluate function for complete mini batch
        for i = 1,#inputs do
          -- estimate f
          local output = softModel:forward(inputs[i])
          local err = criterion:forward(output, targets[i])
          f = f + err
          -- estimate df/dW
          local df_do = criterion:backward(output, targets[i])
          softModel:backward(inputs[i], df_do)
          -- update confusion
          confusion:add(output, targets[i])
        end
        -- normalize gradients and f(X)
        gradParameters:div(#inputs)
        f = f/#inputs
        -- return f and df/dX
        return f,gradParameters
      end
      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
    end
    confusion:zero();
  end
  -- set model to evaluate mode
  --softModel:evalute();
  for t = 1,testData:size() do
    -- disp progress
    xlua.progress(t, testData:size())
    -- get new sample
    local input = testData.data[t]
    if opt.type == 'double' then input = input:double()
    elseif opt.type == 'cuda' then input = input:cuda() end
    local target = testData.labels[t]
    -- test sample
    local pred = softModel:forward(input)
    confusion:add(pred, target)
  end
  return (1.0 - confusion.totalValid);
end