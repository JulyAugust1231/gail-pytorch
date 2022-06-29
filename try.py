import torch
import math
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from models.nets import Discriminator,OneHotDQN,ValueNetwork
from torch import FloatTensor
import numpy as np
state_dim = 2

ep_obs = torch.tensor([[0., 1.], [1., 0.],[0,1]])
ep_acts = torch.tensor([0., 3.,2])
discrete = True
d = Discriminator(state_dim, 4, discrete)
print(d(ep_obs, ep_acts))
ep_costs = (-1) * torch.log(d(ep_obs, ep_acts))\
                    .squeeze().detach()

print(ep_costs)

dqn = OneHotDQN(state_dim, 4)
q_value = dqn(torch.tensor([0., 1.]))
print(q_value)

v = ValueNetwork(state_dim)
print(v.eval())
curr_vals = v(ep_obs).detach()
next_vals = torch.cat(
    (v(ep_obs)[1:], FloatTensor([[0.]]))
).detach()

print('curr_vals',curr_vals)
print('next_vals',next_vals)

ep_costs = (-1) * torch.log(d(ep_obs, ep_acts))\
                    .squeeze().detach()  #在一条轨迹中，每个time step,用discriminator求出的值（输入有state space和action space，输出值)。 故有T维的cost
print('ep_costs',ep_costs)

print(torch.log(d(ep_obs, ep_acts)))

ep_disc_rets = FloatTensor(
                    [sum(ep_costs[i:]) for i in range(3)]
                )
print('ep_disc',ep_disc_rets)

rets = []
rets.append(ep_disc_rets)
print('rets',rets)
rets = torch.cat(rets)
print(rets)

A = [1,2,3,4,5,6]
print(np.mean(A[-10:]))
print(np.mean(A))