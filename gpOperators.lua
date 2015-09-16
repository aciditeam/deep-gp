----------------------------------------------------------------------
--
-- Deep Genetic Programming: Reifying an AI researcher.
--
-- Genetic operators to make the population evolve
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Global variables for operators
MutateConnectionsChance = 0.25      -- pointMutate
PerturbChance = 0.90                -- pointMutate (chance of pure random)
CrossoverChance = 0.75              -- crossover
LinkMutationChance = 2.0            -- linkMutate(false)
NodeMutationChance = 0.50           -- nodeMutate
BiasMutationChance = 0.40           -- linkMutate(true)
StepSize = 5                        -- pointMutate (weight step)
DisableMutationChance = 0.4         -- enableDisableMutate(false)
EnableMutationChance = 0.2          -- enableDisableMutate(true)

----------------------------------------------------------------------
-- Perform crossover between two genomes
function crossover(g1, g2)
	-- Make sure g1 is the best genome
	if g2.error < g1.error then
		tempg = g1
		g1 = g2
		g2 = tempg
	end
  -- Create an empty child
	local child = newGenome()
  -- Create a table of genes in 2nd
	local innovations2 = {}
	for i=1,#g2.genes do
		local gene = g2.genes[i]
		innovations2[gene.innovation] = gene
	end
	-- Select gene either from 1st or 2nd (based on innovation ?)
	for i=1,#g1.genes do
		local gene1 = g1.genes[i]
		local gene2 = innovations2[gene1.innovation]
		if gene2 ~= nil and math.random(2) == 1 and gene2.enabled then
      if containsLink(child.genes, gene2) then
        goto continue
      end
			table.insert(child.genes, copyGene(gene2))
      child.neurons[gene2.into] = g2.neurons[gene2.into];
      child.neurons[gene2.out] = g2.neurons[gene2.out];
		else
      if containsLink(child.genes, gene1) then
        goto continue
      end
			table.insert(child.genes, copyGene(gene1))
      child.neurons[gene1.into] = g1.neurons[gene1.into];
      child.neurons[gene1.out] = g1.neurons[gene1.out];
		end
    ::continue::
	end
	child.maxneuron = math.max(g1.maxneuron,g2.maxneuron)
	-- Keep the mutation rates of the best parent
	for mutation,rate in pairs(g1.mutationRates) do
		child.mutationRates[mutation] = rate
	end
	generateNetwork(child);
	return child
end

----------------------------------------------------------------------
-- Select a random neuron (either any or just non-inputs)
function randomNeuron(genome, nonInput)
	local neurons = {};
  -- Add input neurons (if any is okay)
	if not nonInput then
		table.insert(neurons, 1);
	end
  if nonInput then
    -- Add output neurons
    table.insert(neurons, MaxNodes+1)
  end
  -- Add neurons (based on links)
	for i=2,(genome.maxneuron-1) do
    table.insert(neurons, i)
	end
	local n = math.random(1, #neurons);
	return neurons[n];
end

----------------------------------------------------------------------
-- Check if a given link exists in the network
function containsLink(genes, link)
	for i=1,#genes do
		local gene = genes[i]
		if gene.into == link.into and gene.out == link.out then
			return true
		end
	end
end

----------------------------------------------------------------------
-- Mutate the weights associated with all links
function pointMutate(genome)
	local step = math.random(math.round(genome.mutationRates["step"]));
  -- Parse through the whole genome
	for i=1,step do
		-- Select any non-input random neuron
    local neuron2 = randomNeuron(genome, true)
    -- Either reparametrize or change module
		if math.random() < PerturbChance then
			parametrizeNeuron(genome.neurons[neuron2]);
		else
			genome.neurons[neuron2] = newNeuron('random', genome.neurons[neuron2].idN);
			parametrizeNeuron(genome.neurons[neuron2]);
		end
	end
end

----------------------------------------------------------------------
-- Mutate by creating a new link
function linkMutate(genome, forceOut)
  -- Select any random neuron
	local neuron1 = randomNeuron(genome, false)
  -- Select any non-input random neuron
	local neuron2 = randomNeuron(genome, true)
  -- Create a new link
	local newLink = newGene()
  --Both input nodes
	if neuron1 == 1 and neuron2 == 1 then
		return
	end
  -- Swap if input
	if neuron2 < neuron1 then
		local temp = neuron1
		neuron1 = neuron2
		neuron2 = temp
	end
  -- Create the new link
	newLink.into = neuron2;
	newLink.out = neuron1;
  -- Force a bias (last input is fake +1)
	if forceOut then
		newLink.into = MaxNodes + 1
	end
	-- Do nothing if the link already exists
	if containsLink(genome.genes, newLink) then
		return
	end
  -- Innovation seems to indicate index
	newLink.innovation = newInnovation()
	-- Insert the new gene in the genome
	table.insert(genome.genes, newLink)
  table.insert(genome.neurons[newLink.into].input, newLink.out);
  table.insert(genome.neurons[newLink.out].output, newLink.into);
end

----------------------------------------------------------------------
-- Mutate by creating a new module
function nodeMutate(genome)
	if #genome.genes == 0 then
		return
	end
  -- Do not exceed the max number of nodes
  if genome.maxneuron == MaxNodes then
    return
  end
  -- Select a random link
	local gene = genome.genes[math.random(1,#genome.genes)]
	if not gene.enabled then
		return
	end
  -- Add a new neuron
	genome.maxneuron = genome.maxneuron + 1
  -- Create a new neuron
  genome.neurons[genome.maxneuron] = newNeuron('random', genome.maxneuron);
  -- Disable the link
	gene.enabled = false
	-- The module is simply added in between the current link
	local gene1 = copyGene(gene)
  -- Signal going in the new module
	gene1.out = genome.maxneuron
	gene1.weight = 1.0
	gene1.innovation = newInnovation()
	gene1.enabled = true
	table.insert(genome.genes, gene1)
  -- Signal going out of the new module
	local gene2 = copyGene(gene)
	gene2.into = genome.maxneuron
	gene2.innovation = newInnovation()
	gene2.enabled = true
	table.insert(genome.genes, gene2)
end

----------------------------------------------------------------------
-- Switch the state of a neuron (enabled / disabled)
function enableDisableMutate(genome, enable)
	local candidates = {}
  -- Create a table of candidates with opposite state
	for _,gene in pairs(genome.genes) do
		if gene.enabled == not enable then
			table.insert(candidates, gene)
		end
	end
  -- No candidates of this state
	if #candidates == 0 then
		return
	end
  -- Select a random gene and switch it
	local gene = candidates[math.random(1,#candidates)]
	gene.enabled = not gene.enabled
end

----------------------------------------------------------------------
-- Mutate a genome
function mutate(genome)
  -- Randomly augment or diminish the mutation rates
	for mutation,rate in pairs(genome.mutationRates) do
		if math.random(1,2) == 1 then
			genome.mutationRates[mutation] = 0.95*rate
		else
			genome.mutationRates[mutation] = 1.05263*rate
		end
	end
  -- Mutate its modules (nodes)
	if math.random() < genome.mutationRates["connections"] then
		pointMutate(genome)
	end
	-- Mutate its links (create new links)
	local p = genome.mutationRates["link"]
  -- Create a set of [0,p] links
	while p > 0 do
		if math.random() < p then
			linkMutate(genome, false)
		end
		p = p - 1
	end
	-- Mutate biases links 
	p = genome.mutationRates["bias"]
  -- Create a set of [0,p] links
	while p > 0 do
		if math.random() < p then
			linkMutate(genome, true)
		end
		p = p - 1
	end
	-- Mutate nodes (create new modules)
	p = genome.mutationRates["node"]
  -- Create a set of [0,p] nodes
	while p > 0 do
		if math.random() < p then
			nodeMutate(genome)
		end
		p = p - 1
	end
	-- Enable some links
	p = genome.mutationRates["enable"]
  -- Enable a set of [0,p] links
	while p > 0 do
		if math.random() < p then
			enableDisableMutate(genome, true)
		end
		p = p - 1
	end
  -- Disable some links
	p = genome.mutationRates["disable"]
  -- Disable a set of [0,p] links
	while p > 0 do
		if math.random() < p then
			enableDisableMutate(genome, false)
		end
		p = p - 1
	end
end

----------------------------------------------------------------------
-- Return percentage of genomes that are disjoint
function disjoint(genes1, genes2)
  -- Collect indices for 1st
	local i1 = {}
	for i = 1,#genes1 do
		local gene = genes1[i]
		i1[gene.innovation] = true
	end
  -- Collect indices for 2nd
	local i2 = {}
	for i = 1,#genes2 do
		local gene = genes2[i]
		i2[gene.innovation] = true
	end
  -- Check if genes of 1st are in 2nd
	local disjointGenes = 0
	for i = 1,#genes1 do
		local gene = genes1[i]
		if not i2[gene.innovation] then
			disjointGenes = disjointGenes+1
		end
	end
  -- Check if genes of 2nd are in 1st
	for i = 1,#genes2 do
		local gene = genes2[i]
		if not i1[gene.innovation] then
			disjointGenes = disjointGenes+1
		end
	end
  -- Take the max number of genes
	local n = math.max(#genes1, #genes2)
	-- Return percentage of disjoint
	return disjointGenes / n
end

----------------------------------------------------------------------
-- Check how far weights from two genomes are disjoint
function weights(genome1, genome2)
	local i2 = {}
  genes1 = genome1.genes;
  genes2 = genome2.genes;
	for i = 1,#genes2 do
		local gene = genes2[i]
		i2[gene.innovation] = gene
	end
	local sum = 0
	local coincident = 0
	for i = 1,#genes1 do
		local gene = genes1[i]
		if i2[gene.innovation] ~= nil then
			local gene2 = i2[gene.innovation];
      local sameOut = tonumber(genome1.neurons[gene1.out].name == genome2.neurons[gene2.out].name);
      local sameIn = tonumber(genome1.neurons[gene1.into].name == genome2.neurons[gene2.into].name);
			sum = sum + sameOut + sameIn;
			coincident = coincident + 2;
		end
	end
	return sum / coincident
end

----------------------------------------------------------------------
-- Two species are considered equivalent if genes and weights are within a defined range
function sameSpecies(genome1, genome2)
	local dd = DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
	local dw = DeltaWeights*weights(genome1, genome2) 
	return dd + dw < DeltaThreshold
end

----------------------------------------------------------------------
-- Rank the entire pool of species
function rankGlobally()
	local global = {}
  -- Create table of species
	for s = 1,#pool.species do
		local species = pool.species[s]
		for g = 1,#species.genomes do
			table.insert(global, species.genomes[g])
		end
	end
  -- Sort the table
	table.sort(global, function (a,b)
		return (a.error > b.error)
	end)
  -- Keep the ranks in each
	for g=1,#global do
		global[g].globalRank = g
	end
end

----------------------------------------------------------------------
-- Calculate the average rank over a species
function calculateAverageFitness(species)
	local total = 0
  -- For its set of genome
	for g=1,#species.genomes do
		local genome = species.genomes[g]
		total = total + genome.globalRank
	end
	-- Compute the average rank
	species.averageRank = total / #species.genomes
end

----------------------------------------------------------------------
-- Calculate the average error over the complete pool
function totalAverageFitness()
	local total = 0
	for s = 1,#pool.species do
		local species = pool.species[s]
		total = total + species.averageRank
	end
	return total
end

----------------------------------------------------------------------
-- Cull the current species (keep only best genomes)
function cullSpecies(cutToOne)
  -- Parse through the pool of species
	for s = 1,#pool.species do
		local species = pool.species[s]
		table.sort(species.genomes, function (a,b)
			return (a.error < b.error)
		end)
		-- Remaining elements (default cut by two)
		local remaining = math.ceil(#species.genomes/2)
    -- If specified keep only best
		if cutToOne then
			remaining = 1
		end
    -- Remove corresponding (sorted by error) genomes
		while #species.genomes > remaining do
			table.remove(species.genomes)
		end
	end
end

----------------------------------------------------------------------
-- Breed a new child (crossover and mutate)
function breedChild(species)
	local child = {}
	if math.random() < CrossoverChance then
    -- Either take the crossover of two random genomes
		g1 = species.genomes[math.random(1, #species.genomes)]
		g2 = species.genomes[math.random(1, #species.genomes)]
		child = crossover(g1, g2)
	else
    -- Or simply copy the genome of an existing species
		g = species.genomes[math.random(1, #species.genomes)]
		child = copyGenome(g)
	end
	-- Mutate corresponding child
	mutate(child);
  generateNetwork(child);
	return child
end
