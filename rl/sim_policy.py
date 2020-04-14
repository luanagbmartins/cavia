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
import argparse
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy
from policies.context import ContextEncoder
from sampler import BatchSampler



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
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        return mean, conf_int[0]
    else:
        return mean



def main(config):
    utils.set_seed(config['seed'], cudnn=config['make_deterministic'])

    # initialise tensorboard writer
    
    # subfolders for logging
    method_used = 'experiment'
    num_context_params = str(config['num_context_params']) + '_' 
    output_name = 'adaptation-test_'
    output_name += num_context_params + 'lr=' + str(config['fast_lr']) +'kl_lambda=' + str(config['kl_lambda'])
    output_name += '_' + datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_folder = os.path.join(os.path.join(dir_path, 'logs'), config['env_name'], method_used, output_name)
    writer = SummaryWriter(log_folder)

    sampler = BatchSampler(config['env_name'], 
                           batch_size = config['fast_batch_size'], 
                           num_workers = config['num_workers'],
                           device = config['device'], 
                           seed = config['seed'])

    obs_dim = int(np.prod(sampler.envs.observation_space.shape))
    action_dim = int(np.prod(sampler.envs.action_space.shape))
    reward_dim = 1
    latent_dim = config['num_context_params']
    context_encoder_input_dim = obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 

    encoder = ContextEncoder(
        context_encoder_input_dim,
        context_encoder_output_dim,
        latent_dim,
        hidden_sizes=(context_encoder_output_dim,) * config['num_layers'],
        kl_lambda=config['kl_lambda'],
        device=config['device']
    )
    policy = NormalMLPPolicy(
        obs_dim,
        action_dim,
        hidden_sizes=(config['hidden_size'],) * config['num_layers'],
        num_context_params=config['num_context_params'],
        device=config['device'],
        encoder=encoder
    )


    policy.load_state_dict(torch.load(os.path.join(config['path_to_exp'], 'policy-499.pt')))
    encoder.load_state_dict(torch.load(os.path.join(config['path_to_exp'], 'context-499.pt')))

    # initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    # initialise meta-learner
    metalearner = MetaLearner(sampler, 
                              policy, 
                              baseline, 
                              gamma = config['gamma'], 
                              fast_lr = config['fast_lr'], 
                              tau = config['tau'],
                              device = config['device'])

    test_tasks = sampler.sample_tasks(num_tasks=config['test_batch_size'])
    test_episodes = metalearner.test(test_tasks,
                                     num_steps=config['num_test_steps'],
                                     batch_size=config['test_batch_size'],
                                     halve_lr=config['halve_test_lr'])

    all_returns = total_rewards(test_episodes, interval=False)
    for num in range(config['num_test_steps'] + 1):
        writer.add_scalar('evaluation/rew', all_returns[num], num)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptation test')
    parser.add_argument('--path-to-exp', type=str,
                        help='path to policy save folder')
    parser.add_argument('--make-deterministic', action='store_true',
                        help='make everything deterministic (set cudnn seed; num_workers=1; '
                             'will slow things down but make them reproducible!)')
    parser.add_argument('--test-batch-size', type=int, default=40,
                        help='batch size (number of trajectories) for testing')
    args = parser.parse_args()
    
    with open(os.path.join(args.path_to_exp, 'config.json')) as f:
            exp_params = json.load(f)

    if args.make_deterministic:
        exp_params['num_workers'] = 1

    exp_params['path_to_exp'] = args.path_to_exp
    exp_params['make_deterministic'] = args.make_deterministic
    exp_params['test_batch_size'] = args.test_batch_size
    
    main(exp_params)