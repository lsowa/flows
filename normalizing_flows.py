import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from utils import *

'''
In this example we will first have a look at the workflow of a (very) simple normalizing planar flow model. 
Afterwards we will use the FrEIa framework to easily initialize a more complex normalizing coupling flow model.
'''


# Note: PlanarFlows are only invertible under certain conditions, 
# nevetheless we use them to illustrate the workflow of a simple model. 

# create a class for planar flows
class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # setup learnable pytorch parameters
        self.b = nn.Parameter(torch.Tensor(1, dim))
        self.w = nn.Parameter(torch.Tensor(dim, dim))
        self.u = nn.Parameter(torch.Tensor(1, dim))
        self.init_parameters()
    
    # flow function
    def __call__(self, z):
        linear = F.linear(z, self.w, self.b)
        return z + self.u * torch.tanh(linear)

    # compute log of the Jacobian's determinant
    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.w, self.b)
        psi = F.linear(1 - torch.tanh(f_z) ** 2, self.w)
        det_grad = 1 + torch.mm(psi, self.u.t())
        return torch.log(det_grad.abs() + 1e-9)

    # initialize pytorch parameters
    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

# define main model to wrap flows

class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_flows):
        super().__init__()
        # stack PlanarFlow n_flows-th times
        flows = [ PlanarFlow(dim) for i in range(n_flows) ]
        self.flows = nn.ModuleList(flows)

    # generator function g(z)=y
    def forward(self, z):
        self.log_det = []
        for b in range(len(self.flows)):
            self.log_det.append(self.flows[b].log_abs_det_jacobian(z))
            z = self.flows[b](z)
        return z, self.log_det 


# Initialize and train the NormalizingFlow model
flow = NormalizingFlow(dim=2, n_flows=2)
train(model=flow,
      name='simplemodel',
      iterations=1801, 
      lr=0.001)

# Check the output of the stacked flow components
model_layerwise(flow, name='simplemodel')


'''
Now we are using more complex flows. To do so we use the FrEIA framework.
'''

# use https://github.com/VLL-HD/FrEIA from Visual Learning Lab Heidelberg
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# constructor for a multilayer perceptron to be used as mapping ('s' or 't')
def mlp_constructor(input_dim=2, out_dim=2, hidden_nodes=100):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_nodes),
        nn.ReLU(),
        nn.Linear(hidden_nodes, hidden_nodes),
        nn.ReLU(),
        nn.Linear(hidden_nodes, hidden_nodes),
        nn.ReLU(),
        nn.Linear(hidden_nodes, out_dim)
        )
    return model

# Initialize FrEIA model and append some modules to it
# have a look at the documentation of the layers:
#   https://vll-hd.github.io/FrEIA/_build/html/FrEIA.modules.html
inn = Ff.SequenceINN(2)
for k in range(2):
    inn.append(Fm.NICECouplingBlock, subnet_constructor=mlp_constructor)
    inn.append(Fm.PermuteRandom)

# train FrEIA model
train(model=inn,
      name='freia',
      iterations=2001, 
      lr=0.001,
      scheduler=0.999)

# Check the output of the stacked flow components
model_layerwise(inn, name='freia')

# Check for bijectivity

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
z = pz.sample((10, ))

inn.cpu()
y, _ = inn(z)
z_rev, _ = inn(y, rev=True)

# inverting from the outputs should give the original inputs again
assert torch.max(torch.abs(z_rev - z)) < 1e-5

