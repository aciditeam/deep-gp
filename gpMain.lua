----------------------------------------------------------------------
--
-- Deep Genetic Programming: Reifying an AI researcher.
--
-- Main script file
--    * Parsing options
--    * Loading datas
--    * Initializing
--    * Launching
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'gpUtils'
require 'gpSearch'
require 'gpPopulation'
require 'gpOperators'
require 'deepModels'
require 'deepOptimization'
require 'deepImport'

----------------------------------------------------------------------
-- Parsing command line arguments
if not opt then
   print '* Parsing options';
   cmd = torch.CmdLine();
   cmd:text();
   cmd:text('Deep Genetic Programming: Reifying an AI researcher.');
   cmd:text();
   cmd:text('Options:');
   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra');
   cmd:option('-visualize', true, 'visualize input data and weights during training');
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin');
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in');
   cmd:option('-plot', false, 'live plot');
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS');
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0');
   cmd:option('-batchSize', 100, 'mini-batch size (1 = pure stochastic)');
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)');
   cmd:option('-epochs', 20, 'number of final softmax epochs');
   cmd:option('-momentum', 0, 'momentum (SGD only)');
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs');
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS');
   cmd:text();
   opt = cmd:parse(arg or {});
end

----------------------------------------------------------------------
-- Top-level global variables required
master_stack = {};       -- Current model to evaluate
module_table = {};       -- Possible models (blocks) available
type_array = {};         -- Types of models (blocks) to choose
type_stack = {};         -- Current stack of different types
stack_size = 32;         -- Size of the stack
num_types = 0;           -- Number of models (blocks) available
initial_complexity = 4;  -- Maximal number of modules in the net
error_history = {};      -- Keep track of the errors
error_threshold = 0.015  -- Error stop criterion
max_num_iterations = 100 -- Number of iterations
current_iteration = 1    -- Current iteration

----------------------------------------------------------------------
-- Set the random seed generator
math.randomseed(os.time());

----------------------------------------------------------------------
-- Data import part
print "* Importing data"
if opt.size == 'extra' then
   print '    - Using extra training data'
   trsize = 73257 + 531131
   tesize = 26032
elseif opt.size == 'full' then
   print '    - Using regular, full training data'
   trsize = 73257
   tesize = 26032
elseif opt.size == 'small' then
   print '    - Using reduced training data, for fast experiments'
   trsize = 10000
   tesize = 2000
end
-- Import the data
import_data();
-- Collect knowledge about the data
noutputs = sets["train"].labels:max();
nsamples = sets["train"].data:size(1);
nfeats = sets["train"].data:size(2);
width = sets["train"].data:size(3);
height = sets["train"].data:size(4);
ndims = 4;
ninputs = nfeats*width*height;

----------------------------------------------------------------------
-- Create the lists of function
establish_functions();

----------------------------------------------------------------------
-- Population properties
population = {};
population_size = 100;
-- initialize the population pool
initializePool();
print('After pool');
-- while we haven't reached our number of iterations
while current_iteration <= max_num_iterations do
  print("iteration #" .. current_iteration)
  ----------------------------------------------------------------------
  -- Evaluation of networks and errors
  ----------------------------------------------------------------------
  nbGenomes = 0;
  totalError = 0;
  for pcount = 1,#pool.species do
    -- Current species
    local curSpecies = pool.species[pcount]
    -- initialize the error for this species
    pool.species[pcount].bestError = 1
    for gcount = 1,#curSpecies.genomes do
      -- Retrieve current genome
      local curGenome = curSpecies.genomes[gcount]
      -- Display the network (debug)
      displayGenome(curGenome, gcount);
      -- Evaluate current genome
      curError = evaluateNetwork(curGenome.model, sets["train"], sets["test"]);
      -- Update error stats
      curGenome.error = curError;
      totalError = totalError + curError;
      nbGenomes = nbGenomes + 1;
    end
  end
  -- Compute the iteration statistics
  average_error = total_error / population_size
  print("lowest error : " .. best_error)
  print("average error: " .. average_error)
  -- Keep track of the evolution of the error
  table.insert(error_history, best_error)
  -- Early stop if error is under our threshold and report success
  if (best_error < error_threshold) then
    break;
  end
  ----------------------------------------------------------------------
  -- Evolution of population
  ----------------------------------------------------------------------
  -- Cull the bottom half of each species
  cullSpecies(false);
  -- Rank all species
	rankGlobally();
  -- Remove the stale ones
	removeStaleSpecies();
  -- Re-rank the species
	rankGlobally();
  -- Compute average rank for species
	for s = 1,#pool.species do
		local species = pool.species[s];
		calculateAverageFitness(species);
	end
  -- Remove based on average ranks
	removeWeakSpecies();
  -- Complete average ranks
	local sum = totalAverageFitness();
	local children = {};
  -- Create child from species
  for s = 1,#pool.species do
    local species = pool.species[s]
    breed = math.floor(species.averageFitness / sum * Population) - 1
    for i=1,breed do
      table.insert(children, breedChild(species))
    end
  end
  -- Cull all but the top member of each species
  cullSpecies(true)
  -- Recreate a number of children (crossover and mutate) to match pop size
  while #children + #pool.species < Population do
    local species = pool.species[math.random(1, #pool.species)]
    table.insert(children, breedChild(species))
  end
  -- Add the children to the species
  for c=1,#children do
    local child = children[c]
    addToSpecies(child)
  end
  pool.generation = pool.generation + 1
  writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile))
  current_iteration = current_iteration + 1;
end
print("all done!")
--print_table(population[best_index].program, 1)
displayGenome();