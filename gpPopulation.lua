----------------------------------------------------------------------
--
-- Deep Genetic Programming: Reifying an AI researcher.
--
-- Functions for handling the population of networks
--    * Generate random network and population
--    * Process the networks
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Global variables for operators
Population = 100;
DeltaDisjoint = 2.0;
DeltaWeights = 0.4;
DeltaThreshold = 1.0;
StaleSpecies = 15;
MaxNodes = 200;

----------------------------------------------------------------------
-- Add a module to the pool (that contains the set of modules)
function newInnovation()
	pool.innovation = pool.innovation + 1
	return pool.innovation
end

----------------------------------------------------------------------
-- Create an empty pool of species
function newPool()
	local pool = {};
	pool.species = {};
	pool.generation = 0;
	pool.innovation = noutputs;
	pool.minError = 0;
	return pool;
end

----------------------------------------------------------------------
-- Create a new empty species
function newSpecies()
	local species = {};
	species.bestError = 0;
	species.staleness = 0;
	species.genomes = {};
	species.averageRank = 0;
	return species;
end

----------------------------------------------------------------------
-- Create a new genome (one particular network topology)
function newGenome()
	local genome = {};
	genome.genes = {};
	genome.neurons = {};
  genome.modules = {};
  genome.model = {};
  genome.constructed = false;
  genome.computed = false;
  genome.trained = false;
	genome.error = 0;
	genome.adjustedError = 0;
	genome.maxneuron = 0;
	genome.maxparams = 0;
	genome.globalRank = 0;
	genome.mutationRates = {};
	genome.mutationRates["connections"] = MutateConnectionsChance;
	genome.mutationRates["link"] = LinkMutationChance;
	genome.mutationRates["bias"] = BiasMutationChance;
	genome.mutationRates["node"] = NodeMutationChance;
	genome.mutationRates["enable"] = EnableMutationChance;
	genome.mutationRates["disable"] = DisableMutationChance;
	genome.mutationRates["step"] = StepSize;
	return genome
end

----------------------------------------------------------------------
-- Copy a given genome (network topology)
function copyGenome(genome)
	local genome2 = newGenome();
	for g=1,#genome.genes do
		table.insert(genome2.genes, copyGene(genome.genes[g]));
	end
  for n=1,#genome.neurons do
    table.insert(genome2.neurons, copyNeuron(genome.neurons[n]));
  end
	genome2.maxneuron = genome.maxneuron;
	genome2.mutationRates["connections"] = genome.mutationRates["connections"];
	genome2.mutationRates["link"] = genome.mutationRates["link"];
	genome2.mutationRates["bias"] = genome.mutationRates["bias"];
	genome2.mutationRates["node"] = genome.mutationRates["node"];
	genome2.mutationRates["enable"] = genome.mutationRates["enable"];
	genome2.mutationRates["disable"] = genome.mutationRates["disable"];
	return genome2;
end
----------------------------------------------------------------------
-- Create a basic (single-mutation) genome
function basicGenome()
	local genome = newGenome();
	local innovation = 1;
	genome.maxneuron = 2;
    -- Create an input neuron
  genome.neurons[1] = newNeuron('input', 1);
   -- Create an output neuron
  genome.neurons[MaxNodes+1] = newNeuron('output', MaxNodes+1);
  -- Insert a link between those two
  curLink = newGene();
  curLink.innovation = newInnovation();
  curLink.into = MaxNodes+1;
  curLink.out = 1;
  -- Insert this link to the genes
  table.insert(genome.genes, curLink);
	mutate(genome);
	return genome;
end

----------------------------------------------------------------------
-- Create a random genome with a given max complexity
function randomGenome(maxComplexity)
	local genome = newGenome();
	local innovation = 1;
  genome.maxneuron = 1;
  -- Select a given complexity
  local nbModules = math.min(math.random(maxComplexity), MaxNodes);
  -- Create an input neuron
  genome.neurons[1] = newNeuron('input', 1);
  -- First create a random set of modules
  for i = 1,nbModules do
    genome.maxneuron = genome.maxneuron + 1;
    genome.neurons[genome.maxneuron] = newNeuron('random', genome.maxneuron);
  end
   -- Create an output neuron
  genome.neurons[MaxNodes+1] = newNeuron('output', MaxNodes+1);
  local nbLinks = math.min(math.random(maxComplexity) + nbModules, MaxNodes);
  -- Then create random connexions
  for i = 1,nbLinks do
    curLink = newGene();
    n1 = randomNeuron(genome, true);
    n2 = randomNeuron(genome, false);
    curLink.into = math.max(n1,n2);
    curLink.out = math.min(n1,n2);
    -- Avoid self-link and already existing link
    if (not containsLink(genome.genes, curLink)) and (not (curLink.into == curLink.out)) then
      curLink.innovation = newInnovation();
      -- Insert the link to the genome
      table.insert(genome.genes, curLink);
    end
  end
  -- Mutate the genome
	mutate(genome);
	return genome;
end

----------------------------------------------------------------------
-- Create an empty gene (connection between modules)
function newGene()
	local gene = {};
	gene.into = 0;
	gene.out = 0;
	gene.enabled = true;
	gene.innovation = 0;
	return gene;
end

----------------------------------------------------------------------
-- Perform a copy of the gene (connection between modules)
function copyGene(gene)
	local gene2 = newGene();
	gene2.into = gene.into;
	gene2.out = gene.out;
	gene2.enabled = gene.enabled;
	gene2.innovation = gene.innovation;
	return gene2;
end

----------------------------------------------------------------------
-- Create a new neuron (particular module)
function newNeuron(initType, idN)
	local neuron = {};
	neuron.input = {};
	neuron.output = {};
  neuron.instance = {};
  neuron.parametrized = false;
  neuron.params = {};
  neuron.type = initType;
  neuron.inputDim = 0;
  neuron.outputDim = 0;
  neuron.offset = 0;
  neuron.idN = idN;
  if    initType == 'input'  then 
    neuron.module = module_table[1];
    neuron.parametrized = true;
  elseif initType == 'output' then 
    neuron.module = module_table[2];
    neuron.parametrized = true;
  elseif initType == 'random' then 
    local r = math.random((#module_table)-2)+2;
    neuron.module = module_table[r];
    neuron.parametrized = true;
    for i = 1,neuron.module.params do
      range = neuron.module.paramRange[i];
      min = range[1];
      std = (range[2] - min);
      table.insert(neuron.params, math.random(std) + min);
    end
  else neuron.module = {}; end
  neuron.computed = false;
  neuron.trained = false;
	return neuron;
end

----------------------------------------------------------------------
-- Perform a copy of a neuron (connection between modules)
function copyNeuron(nOrig)
	local neuron = newNeuron('empty', nOrig.idN);
  neuron.instance = nOrig.instance;
  neuron.parametrized = nOrig.parametrized;
  neuron.type = nOrig.type;
  neuron.module = nOrig.module;
  neuron.computed = nOrig.computed;
  neuron.trained = nOrig.trained;
  neuron.params = {};
  for i = 1,#nOrig.params do
    table.insert(neuron.params, nOrig.params[i]);
  end
  neuron.input = {};
  for i = 1,#nOrig.input do
    table.insert(neuron.input, nOrig.input[i]);
  end
  neuron.output = {};
  for i = 1,#nOrig.output do
    table.insert(neuron.output, nOrig.output[i]);
  end
  neuron.inputDim = nOrig.inputDim;
  neuron.outputDim = nOrig.outputDim;
  neuron.offset = nOrig.offset;
	return neuron;
end

----------------------------------------------------------------------
-- Parametrize a given neuron (particular module)
function parametrizeNeuron(neuron)
  neuron.parametrized = true;
  for i = 1,neuron.module.params do
    range = neuron.module.paramRange[i];
    min = range[1];
    std = (range[2] - min);
    table.insert(neuron.params, (math.random(std) - 1) + min);
  end
end

----------------------------------------------------------------------
-- Instantiate the module of a neuron
function instantiateModule(neuron, inMod)
  local modFunc = neuron.module.func;
  local nP = neuron.module.params;
  local params = neuron.params;
  local inst = {};
  if      (nP == 0) then  inst = modFunc()(inMod);
  elseif  (nP == 1) then  inst = modFunc(params[1])(inMod);
  elseif  (nP == 2) then  inst = modFunc(params[1],params[2])(inMod);
  elseif  (nP == 3) then  inst = modFunc(params[1],params[2],params[3])(inMod);
  elseif  (nP == 4) then  inst = modFunc(params[1],params[2],params[3],params[4])(inMod);
  elseif  (nP == 5) then  inst = modFunc(params[1],params[2],params[3],params[4],params[5])(inMod); end
  return inst;
end

----------------------------------------------------------------------
-- Prune network (to ensure its correctness)
function pruneNetwork(genome)
  local newGenes = {};
  -- First prune the links
  for i = 1,#genome.genes do
    local curGene = genome.genes[i];
    curGene.into = math.max(curGene.into, curGene.out)
    curGene.out = math.min(curGene.into, curGene.out)
    if not (curGene.into == curGene.out) then
      if not containsLink(newGenes, curGene) then
        table.insert(newGenes, curGene);
      else
        for g = 1,#newGenes do
          if (newGenes[g].out == curGene.out) and (newGenes[g].into == curGene.into) then
            newGenes[g].enabled = (newGenes[g].enabled or curGene.enabled)
            goto nextGene
          end
        end
        ::nextGene::
      end
    end
  end
  genome.genes = newGenes;
  -- First remove all in/out information from neurons
  for n,curNeuron in pairs(genome.neurons) do
    curNeuron.input = {};
    curNeuron.output = {};
  end
  -- Now retrieve and check neuron connections
  for g = 1,#newGenes do
    local curGene = newGenes[g];
    if curGene.enabled then
      table.insert(genome.neurons[curGene.out].output, curGene.into);
      table.insert(genome.neurons[curGene.into].input, curGene.out);
    end
  end
  -- Now check "dead nodes" (either no input or no output)
  for n,curNeuron in ipairs(genome.neurons) do
    -- Non-input node without input
    if (not (n == 1)) and (#curNeuron.input == 0) then
      curLink = newGene();
      curLink.into = n;
      curLink.out = 1;
      curLink.innovation = newInnovation();
      -- Insert connection to the input
      table.insert(genome.genes, curLink);
      table.insert(curNeuron.input, 1);
      table.insert(genome.neurons[1].output, n);
    end
    -- Non-output node without output
    if (not (n == MaxNodes + 1)) and (#curNeuron.output == 0) then
      curLink = newGene();
      curLink.into = MaxNodes+1;
      curLink.out = n;
      curLink.innovation = newInnovation();
      -- Insert connection to the input
      table.insert(genome.genes, curLink);
      table.insert(curNeuron.output, MaxNodes+1);
      table.insert(genome.neurons[MaxNodes+1].input, n);
    end
  end
end

----------------------------------------------------------------------
-- Helper functions for table
function tableContains(tab, val)
  for i,v in ipairs(tab) do
    if (v == val) then
      return true;
    end
  end
  return false;
end

function tableIndex(tab, val)
  for i,v in ipairs(tab) do
    if (v == val) then
      return i;
    end
  end
  return -1;
end

----------------------------------------------------------------------
-- Generate a new network from a genome (topology)
function generateNetwork(genome)
  -- Prune the network
  pruneNetwork(genome);
  -- Sort the genes (by growing index of module)
	table.sort(genome.genes, function (a,b)
		return (a.out < b.out)
	end)
  printGenome(genome);
  -- Create the network in-between
	for i,curNeuron in ipairs(genome.neurons) do
    local outMod = {};
    local inputMods = {};
    print(i);
    if (i == 1) then
      genome.neurons[1].instance = nn.Identity()();
      genome.neurons[1].inputDim = ninputs;
      genome.neurons[1].outputDim = ninputs;
      goto continue
    end
    curNeuron.inputDim = 0;
    -- List all incoming modules
    print('Adding module '..i..' - '..curNeuron.module.name);
    str = 'Inputs -'
    for n = 1,#curNeuron.input do
      table.insert(inputMods, genome.neurons[curNeuron.input[n]].instance);
      curNeuron.inputDim = curNeuron.inputDim + genome.neurons[curNeuron.input[n]].outputDim;
      str = str..' '..genome.neurons[curNeuron.input[n]].module.name..' ('..curNeuron.input[n]..') -';
    end
    print(str);
    if #inputMods > 1 then
      inMod = nn.JoinTable(1)(inputMods);
    else
      inMod = inputMods;
    end
    -- No dimensionality problem here
    if (curNeuron.module.params == 0) or (not tableContains(curNeuron.module.paramNames, 'inputDimension')) then
      print('Case no dimension');
      curNeuron.instance = instantiateModule(curNeuron, inMod);
      curNeuron.outputDim = curNeuron.inputDim;
    else
      local pID = tableIndex(curNeuron.module.paramNames, 'inputDimension');
      nDim = curNeuron.params[pID];
      if (nDim < curNeuron.inputDim) then
        if (curNeuron.offset == 0) then
          curNeuron.offset = math.random(curNeuron.inputDim - nDim)
        end
        print('Case inferior input - '..nDim..' vs. '..curNeuron.inputDim..' - '..curNeuron.offset);
        curNeuron.instance = instantiateModule(curNeuron, nn.NarrowTable(curNeuron.offset, nDim)(inMod));
      else
        curNeuron.params[pID] = curNeuron.inputDim;
        print('Case superior input - '..nDim..' vs. '..curNeuron.inputDim..' - '..curNeuron.params[pID]);
        curNeuron.instance = instantiateModule(curNeuron, inMod);
      end
      curNeuron.outputDim = curNeuron.params[tableIndex(curNeuron.module.paramNames, 'outputDimension')];
    end
    ::continue::
	end
  -- Finally instatiate output neuron
  curNeuron = genome.neurons[MaxNodes+1];
  inputMods = {};
  for n = 1,#curNeuron.input do
    table.insert(inputMods, genome.neurons[curNeuron.input[n]].instance);
    curNeuron.inputDim = curNeuron.inputDim + genome.neurons[curNeuron.input[n]].outputDim;
  end
  if #inputMods > 1 then
    curNeuron.instance = nn.JoinTable(1)(inputMods);
  else
    curNeuron.instance = nn.Identity()(inputMods);
  end
  curNeuron.outputDim = curNeuron.inputDim;
  -- Modules table
  local inModules = {genome.neurons[1].instance};
  local outModules = {genome.neurons[MaxNodes+1].instance};
  -- Generate the model out of the graph
	genome.model = nn.gModule(inModules, outModules);
  print(genome.model);
  genome.constructed = true;
end

----------------------------------------------------------------------
-- Remove stale species (not showing improvement over several iterations)
function removeStaleSpecies()
	local survived = {}
  -- Parse through the pool of species
	for s = 1,#pool.species do
		local species = pool.species[s]
		-- Rank the current genomes inside the species
		table.sort(species.genomes, function (a,b)
			return (a.error < b.error)
		end)
		if species.genomes[1].error < species.bestError then
      -- If this species has shown improvement
			species.bestError = species.genomes[1].error
			species.staleness = 0
		else
      -- Otherwise the species is "stale" (does not improve over time)
			species.staleness = species.staleness + 1
		end
    -- Check if the staleness is within acceptable range
		if species.staleness < StaleSpecies or species.bestError < pool.maxError then
			table.insert(survived, species)
		end
	end
  -- Keep only survivors species
	pool.species = survived
end

----------------------------------------------------------------------
-- Remove weak species (based on relative ranks)
function removeWeakSpecies()
	local survived = {}
  -- Compute the total rank
	local sum = totalAverageFitness()
	for s = 1,#pool.species do
		local species = pool.species[s]
		breed = math.floor(species.averageRank / sum * Population)
		if breed >= 1 then
			table.insert(survived, species)
		end
	end
  -- Keep only best species
	pool.species = survived
end

----------------------------------------------------------------------
-- Add a child to the species
function addToSpecies(child)
	local foundSpecies = false
  -- Check in the pool
	for s=1,#pool.species do
		local species = pool.species[s]
    -- If we found a similar species, we just add to its set of genomes
		if not foundSpecies and sameSpecies(child, species.genomes[1]) then
			table.insert(species.genomes, child)
			foundSpecies = true
		end
	end
  -- If not, this means a new species
	if not foundSpecies then
		local childSpecies = newSpecies()
		table.insert(childSpecies.genomes, child)
		table.insert(pool.species, childSpecies)
	end
end

----------------------------------------------------------------------
-- Initialize a new pool of species
function initializePool()
	pool = newPool();
	for i=1,Population do
		basic = randomGenome(initial_complexity);
    --basic = basicGenome();
    generateNetwork(basic);
    -- Display the network (debug)
    displayGenome(basic, i);
		addToSpecies(basic);
	end
end