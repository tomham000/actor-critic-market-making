from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from tensorboard import program

from simulated_environment.agent_training.callbacks import TensorboardCallback
from simulated_environment.environments import RLEnvironment
from simulated_environment.config import a2c_log_dir
from hyperparameters import parameters

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', a2c_log_dir])
url = tb.launch()
print(f"Tensorflow listening on {url}")

env = Monitor(RLEnvironment())
eval_env = Monitor(RLEnvironment())

eval_callback = EvalCallback(eval_env, best_model_save_path=a2c_log_dir,
                             log_path=a2c_log_dir, eval_freq=500, n_eval_episodes=5,
                             deterministic=True, render=False)

tensorboard_callback = TensorboardCallback()
callback = CallbackList([tensorboard_callback, eval_callback])
policy_kwargs = dict(net_arch=[10,10])
model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, gamma=1.0, tensorboard_log=a2c_log_dir, device='cuda')
total_timesteps = parameters['timesteps']
model.learn(total_timesteps=total_timesteps, callback=callback)
