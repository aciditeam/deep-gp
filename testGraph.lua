require 'nngraph'
require 'nn'

input = nn.Identity()()
L1 = nn.Tanh()(nn.Linear(10, 20)(input))
L2 = nn.Tanh()(nn.Linear(30, 60)(nn.JoinTable(1)({input, L1})))
L3 = nn.Tanh()(nn.Linear(80, 160)(nn.JoinTable(1)({L1, L2})))
g = nn.gModule({input}, {L3})
