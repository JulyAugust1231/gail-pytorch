
import pickle
import os
import json

import gym

from models.nets import *
from models.gail import *

if __name__ == "__main__":
    path = os.path.abspath(os.path.dirname(__file__))
    expert_path = "C:/Users/SQQQ/OneDrive - Singapore Management " \
                  "University/SMU/Project/5_GAIL/2_Implementation/3_CMDP/gail-pytorch/experts/CartPole-v1/"
    print(path)
    with open('results.pkl','rb') as fo:
        dict_data = pickle.load(fo,encoding='bytes')

    with open(expert_path+'model_config.json') as f:
        expert_config = json.load(f)
    print(expert_config)
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)

    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    expert = Expert(
        state_dim, action_dim, discrete, **expert_config
    ).to(device)

    expert.pi.load_state_dict(
        torch.load(
            os.path.join(expert_path, "policy.ckpt"), map_location=device
        )
    )

    #print(state_dim,action_dim)
    #print(expert.pi.load_state_dict)
    # print(dict_data)
    # print(len(dict_data[1]))
    #c = torch.load(
            #os.path.join(expert_path, "policy.ckpt"), map_location=device)
    #print(c)
    #print(env.observation_space.high)
    #print(env.reset())
    #states = env.observation_space.high
    #act1 = expert.act(states)
    #print(act1)
    #ob, rwd, done, info = env.step(act1)
    #print(ob,rwd,done,info)
    #act2 = expert.act(ob)
    #print(act2)
    #print(env.step(act2))
    with open("C:/Users/SQQQ/OneDrive - Singapore Management University/SMU/Project/5_GAIL/2_Implementation/3_CMDP/gail-pytorch/config.json") as f:
        config = json.load(f)[env_name]
    #print(config)
    model = GAIL(state_dim, action_dim, discrete, config).to(device)
    #print(model.d)
    rwd_iter_means = []
    render = False
    num_steps_per_iter = model.train_config["num_steps_per_iter"]
    horizon = model.train_config["horizon"]
    lambda_ = model.train_config["lambda"]
    gae_gamma = model.train_config["gae_gamma"]
    gae_lambda = model.train_config["gae_lambda"]
    eps = model.train_config["epsilon"]
    max_kl = model.train_config["max_kl"]
    cg_damping = model.train_config["cg_damping"]
    normalize_advantage = model.train_config["normalize_advantage"]
    opt_d = torch.optim.Adam(model.d.parameters())

    exp_rwd_iter = []
    exp_obs = []
    exp_acts = []

    steps = 0
    print(num_steps_per_iter)
    while steps < num_steps_per_iter:
        ep_obs = []
        ep_rwds = []

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

            t += 1
            steps += 1

            if horizon is not None:
                if t >= horizon:
                    done = True
                    break
        print(steps)
        if done:
            exp_rwd_iter.append(np.sum(ep_rwds))
        print('ep_rwds',ep_rwds)
        ep_obs = FloatTensor(np.array(ep_obs))
        ep_rwds = FloatTensor(ep_rwds)

    exp_rwd_mean = np.mean(exp_rwd_iter)
    print('exp_rwd_iter',exp_rwd_iter)
    print(
        "Expert Reward Mean: {}".format(exp_rwd_mean)
    )

    exp_obs = FloatTensor(np.array(exp_obs))
    print('exp_obs',len(exp_obs))
    exp_acts = FloatTensor(np.array(exp_acts))


    for i in range(1):
        rwd_iter = []

        obs = []
        acts = []
        rets = []
        advs = []
        gms = []

        steps = 0
        while steps < 4:
            ep_obs = []
            ep_acts = []
            ep_rwds = []
            ep_costs = []
            ep_disc_costs = []
            ep_gms = []
            ep_lmbs = []

            t = 0
            done = False

            ob = env.reset()

            while not done and steps < 4:
                act = model.act(ob)

                ep_obs.append(ob)
                obs.append(ob)

                ep_acts.append(act)
                acts.append(act)

                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)

                ep_rwds.append(rwd)
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

            ep_obs = FloatTensor(np.array(ep_obs))
            print('ep-obs',ep_obs)
            ep_acts = FloatTensor(np.array(ep_acts))
            print('ep-acts', ep_acts)
            ep_rwds = FloatTensor(ep_rwds)
            # ep_disc_rwds = FloatTensor(ep_disc_rwds)
            ep_gms = FloatTensor(ep_gms)
            ep_lmbs = FloatTensor(ep_lmbs)

            ep_costs = (-1) * torch.log(model.d(ep_obs, ep_acts)) \
                .squeeze().detach()
            print(ep_costs)
            ep_disc_costs = ep_gms * ep_costs
            print(ep_disc_costs)
            ep_disc_rets = FloatTensor(
                [sum(ep_disc_costs[i:]) for i in range(t)]
            )
            ep_rets = ep_disc_rets / ep_gms
            print(ep_disc_rets)
            rets.append(ep_rets)

            model.v.eval()
            curr_vals = model.v(ep_obs).detach()
            print('curr_vals',curr_vals)
            next_vals = torch.cat(
                (model.v(ep_obs)[1:], FloatTensor([[0.]]))
            ).detach()
            ep_deltas = ep_costs.unsqueeze(-1) \
                        + gae_gamma * next_vals \
                        - curr_vals

            ep_advs = FloatTensor([
                ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                for j in range(t)
            ])
            advs.append(ep_advs)

            gms.append(ep_gms)

        rwd_iter_means.append(np.mean(rwd_iter))
        print(
            "Iterations: {},   Reward Mean: {}"
                .format(i + 1, np.mean(rwd_iter))
        )

        obs = FloatTensor(np.array(obs))
        acts = FloatTensor(np.array(acts))
        rets = torch.cat(rets)
        print('rets',rets)
        advs = torch.cat(advs)
        print('advs',advs)
        gms = torch.cat(gms)

        if normalize_advantage:
            advs = (advs - advs.mean()) / advs.std()

        model.d.train()
        print(model.d.train())
        exp_scores = model.d.get_logits(exp_obs, exp_acts)
        print('exp_scores',exp_scores)
        nov_scores = model.d.get_logits(obs, acts)
        print('nov_scores',nov_scores)

        opt_d.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            exp_scores, torch.zeros_like(exp_scores)
        ) \
               + torch.nn.functional.binary_cross_entropy_with_logits(
            nov_scores, torch.ones_like(nov_scores)
        )  # discriminator loss w
        loss.backward()
        opt_d.step()

        model.v.train()
        old_params = get_flat_params(model.v).detach()
        old_v = model.v(obs).detach()
        print('old_params',old_params)
        print('old_v',old_v)


        def constraint():
            return ((old_v - model.v(obs)) ** 2).mean()


        grad_diff = get_flat_grads(constraint(), model.v)
        print('grad_diff',grad_diff,len(grad_diff))

        def Hv(v):
            hessian = get_flat_grads(torch.dot(grad_diff, v), model.v) \
                .detach()

            return hessian


        g = get_flat_grads(
            ((-1) * (model.v(obs).squeeze() - rets) ** 2).mean(), model.v
        ).detach()
        print('g',g,len(g))
        s = conjugate_gradient(Hv, g).detach()

        Hs = Hv(s).detach()
        alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

        new_params = old_params + alpha * s

        set_params(model.v, new_params)

        model.pi.train()
        old_params = get_flat_params(model.pi).detach()
        old_distb = model.pi(obs)
        print(old_params)
        print(old_distb)
        def L():
            distb = model.pi(obs)

            return (advs * torch.exp(
                distb.log_prob(acts)
                - old_distb.log_prob(acts).detach()
            )).mean()


        def kld():
            distb = model.pi(obs)

            if model.discrete:
                old_p = old_distb.probs.detach()
                p = distb.probs

                return (old_p * (torch.log(old_p) - torch.log(p))) \
                    .sum(-1) \
                    .mean()

            else:
                old_mean = old_distb.mean.detach()
                old_cov = old_distb.covariance_matrix.sum(-1).detach()
                mean = distb.mean
                cov = distb.covariance_matrix.sum(-1)

                return (0.5) * (
                        (old_cov / cov).sum(-1)
                        + (((old_mean - mean) ** 2) / cov).sum(-1)
                        - model.action_dim
                        + torch.log(cov).sum(-1)
                        - torch.log(old_cov).sum(-1)
                ).mean()


        grad_kld_old_param = get_flat_grads(kld(), model.pi)
        print('grad_kld_old_param',grad_kld_old_param)

        def Hv(v):
            hessian = get_flat_grads(
                torch.dot(grad_kld_old_param, v),
                model.pi
            ).detach()

            return hessian + cg_damping * v


        g = get_flat_grads(L(), model.pi).detach()

        s = conjugate_gradient(Hv, g).detach()
        Hs = Hv(s).detach()

        new_params = rescale_and_linesearch(
            g, s, Hs, max_kl, L, kld, old_params, model.pi
        )

        disc_causal_entropy = ((-1) * gms * model.pi(obs).log_prob(acts)) \
            .mean()
        grad_disc_causal_entropy = get_flat_grads(
            disc_causal_entropy, model.pi
        )
        new_params += lambda_ * grad_disc_causal_entropy

        set_params(model.pi, new_params)




