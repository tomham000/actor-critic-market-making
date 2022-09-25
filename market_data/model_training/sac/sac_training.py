from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor

from market_data.environment import MMEnvironmentBoxActions
from market_data.hyperparameters import hyperparameters, dir_names
from market_data.model_training.callbacks import TensorboardCallback

if __name__ == '__main__':
    algo_name = 'sac'

    log_dir = dir_names[f'{algo_name}_logs']
    env = Monitor(MMEnvironmentBoxActions(debug=False, algo_name=algo_name))
    eval_env = Monitor(MMEnvironmentBoxActions(debug=False, eval_env=True, algo_name=algo_name,
                                               data_path='./TSLA_clean_test_data.csv'))

    eval_callback = EvalCallback(eval_env, best_model_save_path=f'{log_dir}/best_models/{env.env_id}/',
                                 log_path=log_dir, eval_freq=500,
                                 deterministic=False, render=False)

    tensorboard_callback = TensorboardCallback()
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=dir_names['model_training_dir'] + env.env_id,
                                             name_prefix=f'{algo_name}_model')

    callback = CallbackList([tensorboard_callback, eval_callback, checkpoint_callback])

    policy_kwargs = dict(net_arch=[10, 10])
    model = SAC('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, gamma=hyperparameters['gamma'],
                tensorboard_log=log_dir, device='cuda')
    total_timesteps = hyperparameters['training_steps']
    model.learn(total_timesteps=total_timesteps, callback=callback)

