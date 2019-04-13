import argparse
import os

import joblib
import numpy as np
import tensorflow as tf
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator
from sac.algos import SAC
from sac.misc import tf_utils
from sac.misc.instrument import run_sac_experiment
from sac.misc.sampler import SimpleSampler
from sac.misc.utils import timestamp
from sac.policies import LatentSpacePolicy
from sac.preprocessors import MLPPreprocessor
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction

try:
    import git
    repo = git.Repo(os.getcwd())
    git_rev = repo.active_branch.commit.name_rev
except:
    git_rev = None

COMMON_PARAMS = {
    'seed': 'random',
    'lr': 3e-4,
    'discount': 0.99,
    'target_update_interval': 1,
    'tau': 1e-2,
    'layer_size': 128,
    'batch_size': 128,
    'max_pool_size': 1e6,
    'n_train_repeat': 1,
	'n_initial_exploration_steps': 1000000,
    'epoch_length': 1000,
    'snapshot_mode': 'gap',
    'snapshot_gap': 1000,
    'sync_pkl': True,
    # lsp configs
    'policy_coupling_layers': 2,
    'policy_s_t_layers': 1,
    'policy_scale_regularization': 0.0,
    'action_prior': 'normal',
    'preprocessing_hidden_sizes': None,
    'preprocessing_output_nonlinearity': 'relu',
    'squash': True,
    'reparameterize': True,

    'git_sha': git_rev
}


ENV_PARAMS = {
    'ant-resume-training': {  # 8 DoF
        'prefix': 'ant-resume-training',
        'env_name': 'ant-rllab',
        'max_path_length': 1000,
        'n_epochs': int(4e3 + 1),
        'scale_reward': 3.0,

        'preprocessing_hidden_sizes': (128, 128, 16),
        'policy_s_t_units': 8,

        'snapshot_gap': 1000,

        'behavior_polcy_path': [
            'ant-rllab-real-nvp-final-00-00/itr_6000.pkl',
        ]
    },
    'humanoid-resume-training': {  # 21 DoF
        'prefix': 'humanoid-resume-training',
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': int(1e4 + 1),
        'scale_reward': 3.0,

        'preprocessing_hidden_sizes': (128, 128, 42),
        'policy_s_t_units': 21,

        'snapshot_gap': 2000,

        'behavior_polcy_path': [
            'humanoid-real-nvp-final-01b-00/itr_10000.pkl',
        ]
    },
}

DEFAULT_ENV = 'humanoid-resume-training'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default=DEFAULT_ENV)
    parser.add_argument('--exp_name',type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--behavior_polcy_path', '-p',
                        type=str, default=None)
    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = COMMON_PARAMS
    params.update(env_params)

    if args.mode == 'local':
        trained_policies_base = os.path.join(os.getcwd(), 'sac/policies/trained_policies')
    elif args.mode == 'ec2':
        trained_policies_base = '/root/code/rllab/sac/policies/trained_policies'

    params['behavior_polcy_path'] = [
      os.path.join(trained_policies_base, p)
      for p in params['behavior_polcy_path']
    ]

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def load_behavior_polcy(policy_path):
    with tf_utils.get_default_session().as_default():
        with tf.variable_scope("behavior_polcy", reuse=False):
            snapshot = joblib.load(policy_path)

    policy = snapshot["policy"]

    return policy


RLLAB_ENVS = {
    'ant-rllab': AntEnv,
    'humanoid-rllab': HumanoidEnv
}


def run_experiment(variant):
    behavior_polcy = load_behavior_polcy(
        policy_path=variant['behavior_polcy_path'])

    env_args = {
        name.replace('env_', '', 1): value
        for name, value in variant.items()
        if name.startswith('env_') and name != 'env_name'
    }
    if 'rllab' in variant['env_name']:
        EnvClass = RLLAB_ENVS[variant['env_name']]
    else:
        raise NotImplementedError

    env = normalize(EnvClass(**env_args))
    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=variant['max_pool_size'],
    )

    sampler = SimpleSampler(
        max_path_length=variant['max_path_length'],
        min_pool_size=variant['max_path_length'],
        batch_size=variant['batch_size']
    )

    base_kwargs = dict(
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_train_repeat=variant['n_train_repeat'],
	    n_initial_exploration_steps=variant['n_initial_exploration_steps'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
        sampler=sampler
    )

    M = variant['layer_size']
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    preprocessing_hidden_sizes = variant.get('preprocessing_hidden_sizes')
    observations_preprocessor = (
        MLPPreprocessor(env_spec=env.spec,
                        layer_sizes=preprocessing_hidden_sizes,
                        name='high_level_observations_preprocessor')
        if preprocessing_hidden_sizes is not None
        else None
    )

    policy_s_t_layers = variant['policy_s_t_layers']
    policy_s_t_units = variant['policy_s_t_units']
    s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

    bijector_config = {
        "scale_regularization": 0.0,
        "num_coupling_layers": variant['policy_coupling_layers'],
        "translation_hidden_sizes": s_t_hidden_sizes,
        "scale_hidden_sizes": s_t_hidden_sizes,
    }

    policy = LatentSpacePolicy(
        env_spec=env.spec,
        mode="train",
        squash=variant['squash'],
        bijector_config=bijector_config,
        reparameterize=variant['reparameterize'],
        q_function=qf1,
        observations_preprocessor=observations_preprocessor,
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
	    initial_exploration_policy=behavior_polcy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        vf=vf,

        lr=variant['lr'],
        scale_reward=variant['scale_reward'],
        discount=variant['discount'],
        tau=variant['tau'],
        target_update_interval=variant['target_update_interval'],
        action_prior=variant['action_prior'],

        save_full_state=False,
    )

    algorithm.train()

def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        if variant['seed'] == 'random':
            variant['seed'] = np.random.randint(1, 100)
        print("Experiment: {}/{}".format(i, num_experiments))
        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = (variant['prefix']
                           + '-' + args.exp_name
                           + '-' + str(i).zfill(2))

        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

if __name__ == '__main__':
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
