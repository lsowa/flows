import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from utils import *

# Flow class
''' Note: PlanarFlows are only invertible under certain conditions, 
nevetheless we use them to illustrate the workflow of a simple model '''

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

# Main Model to wrap flows

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
      iterations=1801, 
      lr=0.001)

# Check the output of the stacked flow components
model_layerwise(flow, type='simple')


######################
# Freia Model
######################

# use https://github.com/VLL-HD/FrEIA from Visual Learning Lab Heidelberg
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# Initialize FrEIA model and append some modules to it
inn = Ff.SequenceINN(2)
for k in range(8):
    inn.append(Fm.NICECouplingBlock, subnet_constructor=mlp_constructor)
    inn.append(Fm.PermuteRandom)

# train FrEIA model
train(model=inn,
      iterations=2001, 
      lr=0.001,
      scheduler=0.999)

# Check the output of the stacked flow components
model_layerwise(inn, type='freia')

# Check for bijectivity

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
z = pz.sample((int(10), ))

inn.cpu()
y, _ = inn(z)
z_rev, _ = inn(y, rev=True)

# inverting from the outputs should give the original inputs again
assert torch.max(torch.abs(z_rev - z)) < 1e-5

