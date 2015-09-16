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

require 'nngraph'
require 'graph'
require 'nn'
--require 'cunn'

-- Sleep function
function sleep(n)
    os.execute("sleep " .. tonumber(n))
end

-- Performs a recursive copy of a table
function deepcopy(object)
    local lookup_table = {}
    -- Main function
    local function _copy(object)
      if type(object) ~= "table" then
          return object
      elseif lookup_table[object] then
          return lookup_table[object]
      end
      local new_table = {}
      lookup_table[object] = new_table
      for index, value in pairs(object) do
          new_table[_copy(index)] = _copy(value)
      end
      return setmetatable(new_table, getmetatable(object))
    end
    return _copy(object)
end

-- Pretty print for a table (with indent)
function print_table(theTable, indent)
    local iString = ""
    for index = 1, indent do
        iString = iString .. "-"
    end
    -- walk all the topmost values in the table
    for k,v in pairs(theTable) do
        print(iString ,k ,v)
        if type(v) == "table" then
			print_table(v, indent + 1)
		end
	end
end

-- Print a complete network
function print_network(theProgram)
	print("Network:");
	for counter = 1, table.getn(theProgram) do
		print(theProgram[counter].name);
		if counter < table.getn(theProgram) then
      print(' | ');
      print(' v ');
    end
	end
	print("Program Stack Top")
end


function writeFile(filename)
        local file = io.open(filename, "w")
	file:write(pool.generation .. "\n")
	file:write(pool.maxFitness .. "\n")
	file:write(#pool.species .. "\n")
        for n,species in pairs(pool.species) do
		file:write(species.topFitness .. "\n")
		file:write(species.staleness .. "\n")
		file:write(#species.genomes .. "\n")
		for m,genome in pairs(species.genomes) do
			file:write(genome.fitness .. "\n")
			file:write(genome.maxneuron .. "\n")
			for mutation,rate in pairs(genome.mutationRates) do
				file:write(mutation .. "\n")
				file:write(rate .. "\n")
			end
			file:write("done\n")
			
			file:write(#genome.genes .. "\n")
			for l,gene in pairs(genome.genes) do
				file:write(gene.into .. " ")
				file:write(gene.out .. " ")
				file:write(gene.weight .. " ")
				file:write(gene.innovation .. " ")
				if(gene.enabled) then
					file:write("1\n")
				else
					file:write("0\n")
				end
			end
		end
        end
        file:close()
end

function savePool()
	local filename = forms.gettext(saveLoadFile)
	writeFile(filename)
end

-- Print a genome
function printGenome(genome)
  -- First an old-school print
	local cells = {}
	local i = 1
	local cell = {}
  print('Modules');
	for n,neuron in pairs(genome.neurons) do
		print('Id:'..n..' - '..neuron.module.name);
	end
  print('Connections');
	for _,gene in pairs(genome.genes) do
		if gene.enabled then
			print(' '..gene.out..' -> '..gene.into);
		end
	end
	for mutation,rate in pairs(genome.mutationRates) do
		print(' m:'..mutation..' = '..rate);
  end
end

-- Display a genome
function displayGenome(genome, idx)
  -- First an old-school print
	local cells = {}
	local i = 1
	local cell = {}
  print('Modules');
	for n,neuron in pairs(genome.neurons) do
		print('Id:'..n..' - '..neuron.module.name);
	end
  print('Connections');
	for _,gene in pairs(genome.genes) do
		if gene.enabled then
			print(' '..gene.out..' -> '..gene.into);
		else
      print(' '..gene.out..' /> '..gene.into);
    end
	end
	for mutation,rate in pairs(genome.mutationRates) do
		print(' m:'..mutation..' = '..rate);
  end
  -- Now a fancy print
  local g = genome.model;
  g:float();
  indata = torch.rand(ninputs):float();
  tmpInData = {};
  for i = 1,#genome.neurons[1].output do
    table.insert(tmpInData, indata);
  end
  g:forward(indata);
  --graph.dot(g.fg, 'Forward Graph', '/tmp/fg_'..idx);
end