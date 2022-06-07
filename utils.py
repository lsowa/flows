import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

# Target distribution
def density_ring(z):
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    norm = torch.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
    return torch.exp(-u)

# Train function
def train(model, iterations=2001, 
          lr=0.001, 
          scheduler=None, 
          device=torch.device('cpu')):
    
    pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
    # Setup for Plots
    id_figure = 2
    fig = plt.figure(figsize=(16, 18))
    plt.subplot(3,4,1)
    x = np.linspace(-5, 5, 2000)
    z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)
    z = np.reshape(z, [z.shape[0] * z.shape[1], -1])
    plt.hexbin(z[:,0], z[:,1], C=density_ring(torch.Tensor(z)).numpy().squeeze(), cmap='rainbow')
    plt.title('Target density', fontsize=15)

    loss_list = []
    jac_list = []

    model.to(device)
    # Main training loop
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, scheduler)
    for iter in range(iterations):
        optimizer.zero_grad()
        # Draw a sample batch from Normal
        z = pz.sample((512, ))
        z = z.to(device)
        # Evaluate flow of transforms
        y, log_jacobians = model(z)
        loss_v = loss(density_ring, y, log_jacobians)
        loss_v.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        
        # safe loss
        loss_list.append(loss_v.cpu().detach().numpy())
        #jac_list.append(log_jacobians.detach().numpy())
        # plot results
        if (iter % int(iterations/10.) == 0):
            print('Iter. {} Loss: {:.5f}'.format(iter, loss_v.item()))
            # Draw random samples
            z = pz.sample((int(1e5), ))
            z = z.to(device)
            # Evaluate flow and plot
            y, _ = model(z)
            y = y.cpu().detach().numpy()
            plt.subplot(3,4,id_figure)
            im = plt.hexbin(y[:,0], y[:,1], cmap='rainbow')
            plt.title('Iteration {}'.format(iter), fontsize=15)
            id_figure += 1
    plt.show()    
    plt.plot(loss_list)
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.show()

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

def loss(density, y, log_jacobians):
    if type(log_jacobians)==list:
        log_jacobians = sum(log_jacobians)
    sum_of_log_jacobians = log_jacobians
    return torch.abs((-sum_of_log_jacobians - torch.log(density(y)+1e-9)).mean())

def model_layerwise(model, type):
    if type == 'freia':
        length = len(model)
    elif type=='simple':
        length = len(model.flows)
    else:
        raise ValueError('Please use "freia" or "simple" for model type')
    model.cpu()
    columns = int(np.sqrt(length)) + 1
    fig, ax = plt.subplots(columns, columns)
    fig.set_figheight(3*columns)
    fig.set_figwidth(4*columns)
    pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
    z = pz.sample((int(1e5), ))
    z = z.detach().numpy()
    ax.flatten()[0].hexbin(z[:,0], z[:,1], cmap='rainbow')
    ax.flatten()[0].set_title('Input distribution', fontsize=10)
    for nth_flow in range(length):
        if type == 'freia':
            z = model[nth_flow]((torch.tensor(z),torch.tensor(z)))[0][0]
        elif type=='simple':
            z = model.flows[nth_flow](torch.tensor(z))
        z = z.detach().numpy()
        ax.flatten()[nth_flow+1].hexbin(z[:,0], z[:,1], cmap='rainbow')
        ax.flatten()[nth_flow+1].set_title('Output of layer {}'.format(nth_flow), fontsize=10)
    for axis in ax.flatten()[length+1:]:
        axis.set_axis_off()







