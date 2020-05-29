import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from rl_utils.optimization import conjugate_gradient
from rl_utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)

import matplotlib.pyplot as plt


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('grad_flow.png')


class MetaLearner(object):
    '''Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    '''

    def __init__( self, sampler, agent, baseline, use_information_bottleneck,
                  kl_lambda=0.01, gamma=0.95, fast_lr=0.5, tau=1.0, device='cpu'):

        self.sampler = sampler
        self.agent = agent
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

        self.kl_lambda = kl_lambda
        self.use_ib = use_information_bottleneck


    def inner_loss(self, episodes, policy_outputs):
        '''Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        '''

        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        # pi = self.agent.policy(episodes.observations)
        pi = policy_outputs

        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)

        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)

        return loss

    def adapt(self, episodes, context, first_order=False, lr=None):
        '''Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        '''

        if lr is None:
            lr = self.fast_lr

        self.agent.policy.zero_grad()
        self.agent.context_encoder.zero_grad()

        policy_outputs = self.agent(episodes.observations, context)

        if self.use_ib:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
        
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)

        # Get the loss on the training episodes
        reinforce_loss = self.inner_loss(episodes, policy_outputs)
        
        # Encoder loss
        loss = kl_loss+reinforce_loss if self.use_ib else reinforce_loss 

        # Get the new parameters after a one-step gradient update
        self.agent.context_encoder.update_params(loss, lr, first_order, params=self.agent.context_encoder.parameters())
        
        self.agent.policy.zero_grad()
        self.agent.detach_z()
        self.agent.infer_posterior(context)

        return (loss.item(), kl_loss.item() if self.use_ib else 0, reinforce_loss.item())

    def sample(self, tasks, first_order=False):
        '''Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`. 
        '''

        episodes, losses, context, context_log_info = [], [], [], []
        for task in tasks:

            self.sampler.reset_task(task)
            self.agent.clear_z()

            train_episodes, context_task = self.sampler.sample(self.agent, gamma=self.gamma)
           
            # inner loop (this only updates the context parameters)
            loss = self.adapt(train_episodes, context_task, first_order=first_order)

            # rollouts after inner loop update
            valid_episodes, context_task = self.sampler.sample(self.agent, gamma=self.gamma)
            
            context_z = self.agent.get_z()
            episodes.append((train_episodes, valid_episodes))
            losses.append(loss)
            context.append(context_task)
            context_log_info.append(context_z)

        return episodes, context, context_log_info, losses

    def test(self, tasks, num_steps, batch_size, halve_lr, get_trajectory=False):
        '''Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.batchsize
        '''

        # Save the old parameters
        old_params = parameters_to_vector(self.agent.context_encoder.parameters())

        episodes_per_task = []
        for task in tasks:

            # reset context params and task
            self.sampler.reset_task(task)
            self.agent.clear_z()

            # gather some initial experience and log performance
            test_episodes, context_task = self.sampler.sample(self.agent, gamma=self.gamma, batch_size=batch_size)

            # initialise list which will log all rollouts for the current task
            curr_episodes = [test_episodes]

            for i in range(1, num_steps + 1):
                lr = self.fast_lr

                # inner-loop update
                self.adapt(test_episodes, context_task, first_order=True, lr=lr)

                # get new rollouts
                test_episodes, context_task = self.sampler.sample(self.agent, gamma=self.gamma, batch_size=batch_size)
                curr_episodes.append(test_episodes)

            episodes_per_task.append(curr_episodes)

            # reset encoder parameters
            vector_to_parameters(old_params, self.agent.context_encoder.parameters())

        self.agent.clear_z()
        return episodes_per_task

    def kl_divergence(self, episodes, context, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), context_task, old_pi in zip(episodes, context, old_pis):

            # this is the inner-loop update
            self.agent.infer_posterior(context_task)
            pi = self.agent.policy(valid_episodes.observations)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, context, damping=1e-2):
        '''Hessian-vector product, based on the Perlmutter method.'''

        def _product(vector):
            kl = self.kl_divergence(episodes, context)
            grads = torch.autograd.grad(kl, self.agent.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.agent.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self, episodes, context, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), context_task, old_pi in zip(episodes, context, old_pis):

            # do inner-loop update:
            self.agent.infer_posterior(context_task)
            with torch.set_grad_enabled(old_pi is None):

                # get action values after inner-loop update
                pi = self.agent.policy(valid_episodes.observations)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = ( pi.log_prob(valid_episodes.actions)
                              - old_pi.log_prob(valid_episodes.actions) )
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
                kls.append(kl)

        return torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(kls, dim=0)), pis

    def step(self, episodes, context, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        '''Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        '''

        self.agent.policy.zero_grad()
        old_loss, _, old_pis = self.surrogate_loss(episodes, context)

        # this part will take higher order gradients through the inner loop:
        grads = torch.autograd.grad(old_loss, self.agent.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes, context, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.agent.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.agent.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, context, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            print('No update?')
            print('Improve = ' + str(improve.item()) + ' | KL = ' + str(kl.item()))
            vector_to_parameters(old_params, self.agent.policy.parameters())

        return loss

    def to(self, device, **kwargs):
        self.baseline.to(device, **kwargs)
        self.agent.to(device, **kwargs)
        self.agent.policy.to(device, **kwargs)
        self.agent.context_encoder.to(device, **kwargs)
        self.device = device
