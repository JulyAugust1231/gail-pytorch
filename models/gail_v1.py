import numpy as np
import torch

from torch.nn import Module

from models.nets import PolicyNetwork, ValueNetwork, Discriminator, OneHotDQN
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch
PIT = 3

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)

        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

        self.cost_model = ValueNetwork(self.state_dim) # for cost

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action




    def train(self, env, expert, render=False):
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]
        lamda2_ = self.train_config['cost_penalty']
        opt_d = torch.optim.Adam(self.d.parameters())

        #####################################################################################
        ## calculate for expert trajectories
        exp_rwd_iter = []
        exp_constraints_iter = []
        exp_obs = []
        exp_acts = []

        steps = 0
        while steps < num_steps_per_iter:
            ep_obs = []
            ep_rwds = []
            ep_constraints = []
            t = 0
            done = False

            ob = env.reset()

            while not done and steps < num_steps_per_iter:
                act = expert.act(ob)

                ep_obs.append(ob)
                exp_obs.append(ob)
                exp_acts.append(act)

                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)

                ep_rwds.append(rwd)
                ep_constraints.append(info['pit'])

                t += 1
                steps += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break

            if done:
                exp_rwd_iter.append(np.sum(ep_rwds))
                exp_constraints_iter.append(np.sum(ep_constraints))

            ep_obs = FloatTensor(np.array(ep_obs))
            ep_rwds = FloatTensor(ep_rwds)

        exp_rwd_mean = np.mean(exp_rwd_iter)
        exp_cost_mean = np.mean(exp_constraints_iter)
        print(
            "Expert Reward Mean: {}".format(exp_rwd_mean),
            "Expert Cost Mean: {}".format(exp_cost_mean)
        )

        exp_obs = FloatTensor(np.array(exp_obs))
        exp_acts = FloatTensor(np.array(exp_acts))
        #################################################################################

        rwd_iter_means = []
        cost_iter_means = []
        for i in range(num_iters):
            rwd_iter = []
            cost_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []
            constraint_rets = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []
                ep_constraints = []

                t = 0
                done = False

                ob = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = self.act(ob)

                    ep_obs.append(ob)
                    obs.append(ob)

                    ep_acts.append(act)
                    acts.append(act)

                    if render:
                        env.render()
                    ob, rwd, done, info = env.step(act)

                    ep_rwds.append(rwd)
                    ep_constraints.append(info['pit'])
                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                if done:
                    rwd_iter.append(np.sum(ep_rwds))
                    cost_iter.append(np.sum(ep_constraints))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                    .squeeze().detach()  #在一条轨迹中，每个time step,用discriminator求出的值（输入有state space和action space，输出值)。 故有T维的cost
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)]
                ) #从每个t开始到最后求和ep_disc_costs


                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                ep_constraints = torch.tensor(ep_constraints)
                ep_constraints_rets = FloatTensor(
                    [sum(ep_constraints[i:]) for i in range(t)]
                ) # for every time t, collect costs from t to end.
                constraint_rets.append(ep_constraints_rets)

                ##### use value
                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                #c_q_values = self.cost_model(ep_obs).detach()   # add cost model here
                #cost_ = env.maze[PIT]
                #cost_ = torch.from_numpy(cost_)
                #cost_ = cost_.unsqueeze(-1)

                #######################################################################################################
                self.cost_model.eval()
                curr_costs = self.cost_model(ep_obs).detach()
                next_costs = torch.cat(
                    (self.cost_model(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                #######################################################################################################

                ep_deltas = ep_costs.unsqueeze(-1)\
                    + gae_gamma * next_vals\
                    - curr_vals - lamda2_*(next_costs - curr_costs)# - lamda2_ * (c_q_values-cost_) #cost goes here

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))
            cost_iter_means.append(np.mean(cost_iter))
            print(
                "Iterations: {},   Reward Mean: {},  Cost Mean: {}"
                .format(i + 1, np.mean(rwd_iter), np.mean(cost_iter))
            )

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)  # 每个轨迹算出的deicriminator的值,几个time step,几维
            advs = torch.cat(advs)  # 每个轨迹算出的value network的值,几个time step,几维
            gms = torch.cat(gms)
            constraint_rets = torch.cat(constraint_rets)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            ############################################################################################################
            #### update discriminator parameters, no update
            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )        # discriminator loss w
            loss.backward()
            opt_d.step()
            ############################################################################################################
            # update value network parameters
            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)
            ############################################################################################################
            # update cost value network parameters
            self.cost_model.train()
            old_params = get_flat_params(self.cost_model).detach()
            old_cost = self.cost_model(obs).detach()

            def constraint():
                return ((old_cost - self.cost_model(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.cost_model)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.cost_model)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.cost_model(obs).squeeze() - constraint_rets) ** 2).mean(), self.cost_model
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.cost_model, new_params)
            ############################################################################################################

            # update policy network parameters
            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()     # 这里是 gradient of log_pi(a|s)

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )  # max_kl: hyper-parameter,

            #############################################################################
            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()  # H(pi), casual entropy
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )  # 计算casual entropy的gradient
            new_params += lambda_ * grad_disc_causal_entropy
            ## 这一块计算casual entropy的gradient,不用改
            ##############################################################################
            set_params(self.pi, new_params)


        return exp_rwd_mean, exp_cost_mean,rwd_iter_means,cost_iter_means
