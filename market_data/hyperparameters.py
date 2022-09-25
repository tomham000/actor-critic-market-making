hyperparameters = {
    'learning_rate': 0.00003,
    'learning_rate_decay': 1-1/100_000,
    'min_learning_rate': 0,
    'gamma': 0.999,
    'epsilon': 0.3,
    'epsilon_decay': 1 - 1/1000,
    'position_penalty': 0.000001,
    'trade_rebate': 0.15,
    'max_book_depth': 5,
    'total_book_depth': 5,
    'event_lookback_steps': 5,
    'env_count': 10,
    'reward_steps': 16,
    'environment_length': 20_000,
    'evaluation_episodes': 10,
    'total_episodes': 2000,
    'max_abs_position': 100,
    'max_spread_ticks': 25,
    'artifical_flow_rate': 0.07,
    'training_steps': 2_000_000
}

dir_names = {
    'env_data': './env_data/',
    'market_data': './tsla_data/',
    'ppo_logs': './ppo_logs/',
    'td3_logs': './td3_logs/',
    'sac_logs': './sac_logs/',
    'ddpg_logs': './ddpg_logs/',
    'model_training_dir': './models/',
    'simulations': './simulations/',
    'parameters': './parameters/',
}
