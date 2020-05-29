import datetime
import json
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import scipy.stats as st
import torch
from tensorboardX import SummaryWriter

import utils
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy
from policies.encoder_mlp import EncoderMlp
from sampler import BatchSampler
from agent import Agent


def get_returns(episodes_per_task):

    # sum up for each rollout, then take the mean across rollouts
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # result will be: num_evals * num_updates
        returns.append(torch.stack(curr_returns, dim=1))

    # result will be: num_tasks * num_evals * num_updates
    returns = torch.stack(returns)
    returns = returns.reshape((-1, returns.shape[-1]))

    return returns


def total_rewards(episodes_per_task, interval=False):

    returns = get_returns(episodes_per_task).cpu().numpy()
    mean = np.mean(returns, axis=0)
    
    # Endpoints of the range that contains alpha percent of the distribution
    # interval(alpha, df, loc=0, scale=1)
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = [mean + critval * st.sem(returns, axis=0) / np.sqrt(len(returns)) for critval in conf_int]
    if interval:
        return mean, conf_int
    else:
        return mean


def main(args):

    print('starting....')
    utils.set_seed(args.seed, cudnn=args.make_deterministic)

    continuous_actions = (args.env_name in [ 'AntVel-v1', 
                                             'AntDir-v1',
                                             'AntPos-v0', 
                                             'HalfCheetahVel-v1', 
                                             'HalfCheetahDir-v1',
                                             '2DNavigation-v0' ])

    # subfolders for logging
    method_used = 'experiment'
    num_context_params = str(args.num_context_params) + '_' 
    output_name = num_context_params + 'lr=' + str(args.fast_lr) + 'kl-lambda=' + str(args.kl_lambda) + 'tau=' + str(args.tau)
    output_name += '_' + datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_folder = os.path.join(os.path.join(dir_path, 'logs'), args.env_name, method_used, output_name)
    save_folder = os.path.join(os.path.join(dir_path, 'saves'), output_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # initialise tensorboard writer
    writer = SummaryWriter(log_folder)

    # save config file
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    with open(os.path.join(log_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler( args.env_name, batch_size=args.fast_batch_size,
                            num_workers=args.num_workers, device=args.device, seed=args.seed )

    obs_dim = int(np.prod(sampler.envs.observation_space.shape))
    action_dim = int(np.prod(sampler.envs.action_space.shape))
    reward_dim = 1
    latent_dim = args.num_context_params
    context_encoder_input_dim = obs_dim*2 + action_dim + reward_dim if args.use_next_obs_in_context else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim*2 if args.use_information_bottleneck else latent_dim

    if continuous_actions:
        context_encoder = EncoderMlp(
            context_encoder_input_dim,
            context_encoder_output_dim,
            hidden_sizes=(context_encoder_input_dim,) * args.context_num_layers,
        )
        policy = NormalMLPPolicy(
            obs_dim,
            action_dim,
            device=args.device,
            hidden_sizes=(args.hidden_size,) * args.num_layers,
            num_context_params=args.num_context_params
        )
        agent = Agent(
            latent_dim=latent_dim,
            context_encoder=context_encoder,
            policy=policy,
            device=args.device,
            use_information_bottleneck=args.use_information_bottleneck,
            use_next_obs_in_context=args.use_next_obs_in_context
        )
    else:
        raise NotImplementedError

    # initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    # initialise meta-learner
    metalearner = MetaLearner( sampler, agent, baseline, args.use_information_bottleneck, kl_lambda=args.kl_lambda,
                               gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau, device=args.device )

    print('Policy\n', policy)
    print('Context Encoder\n', context_encoder)

    for batch in range(args.num_batches):
        # get a batch of tasks
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)

        # do the inner-loop update for each task
        # this returns training (before update) and validation (after update) episodes
        episodes, context, context_log_info, inner_losses = metalearner.sample(tasks, first_order=args.first_order)

        # take the meta-gradient step
        outer_loss = metalearner.step( episodes, context, max_kl=args.max_kl, cg_iters=args.cg_iters,
                                       cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                                       ls_backtrack_ratio=args.ls_backtrack_ratio )

        # ----- logging -----
        curr_returns = total_rewards(episodes, interval=False)
        print('   return after update: ', curr_returns[1])

        # tensorboard
        writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)

        writer.add_scalar('running_returns/before_update', curr_returns[0], batch)
        writer.add_scalar('running_returns/after_update', curr_returns[1], batch)

        writer.add_scalar('loss/inner_rl', np.mean(inner_losses, axis=0)[0], batch)
        writer.add_scalar('loss/kl_loss', np.mean(inner_losses, axis=0)[1], batch)
        writer.add_scalar('loss/reinforce_loss', np.mean(inner_losses, axis=0)[2], batch)
        writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        # inference
        if args.use_information_bottleneck:
            writer.add_scalar('posterior/z_means', np.mean(context_log_info, axis=0)[0], batch)
            writer.add_scalar('posterior/z_vars', np.mean(context_log_info, axis=0)[1], batch)
        else:
            writer.add_scalar('posterior/z_means', np.mean(context_log_info, axis=0), batch)

        # ----- evaluation -----
        # evaluate for multiple update steps
        if batch % args.test_freq == 0:
            test_tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            test_episodes = metalearner.test(test_tasks, num_steps=args.num_test_steps,
                                             batch_size=args.test_batch_size, halve_lr=args.halve_test_lr)
            all_returns = total_rewards(test_episodes, interval=False)
            for num in range(args.num_test_steps + 1):
                writer.add_scalar('evaluation_rew/avg_rew ' + str(num), all_returns[num], batch)

            print('   inner RL loss:', np.mean(inner_losses, axis=0)[0])
            print('   outer RL loss:', outer_loss.item())

        # ----- save policy network -----
        with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)
        with open(os.path.join(save_folder, 'context-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(context_encoder.state_dict(), f)



if __name__ == '__main__':
    args = parse_args()
    main(args)
