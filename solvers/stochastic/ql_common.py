COMMON_RUN_PARAMS = {
    "num_episodes": 1000000,
    "episode_length": 50,
    "num_replay_passes": 10,
    "exploration_prob": 0.95,
    "learning_rate": 0.1,
    "decay_rate": 0.51,
    "lr_decay_scale": 0.0000001,
    "log_every": 1000,
    "reward_ma_window": 2000,
    "td_ma_window": 2000,
    "random_seed": 0,
    "verbose": False,
}
