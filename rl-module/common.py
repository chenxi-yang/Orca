import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import models
from replay_buffer import (
    BootstrapIterator,
    ReplayBuffer,
    SequenceTransitionIterator,
    SequenceTransitionSampler,
    TransitionIterator,
)


def create_one_dim_tr_model(
    s_dim,
    a_dim,
    model_dir,
    model_normalization,
):
    """
    Creates a 1-D transition reward model from a given configuration.
    """
    # TODO: use fixed configs for now
    in_size = s_dim + a_dim
    out_size = s_dim + 1 # state + reward

    model = models.SymGaussianMLP(
        in_size=in_size,
        out_size=out_size,
        device="/cpu",
        num_layers=4,
        ensemble_size=1,
        hid_size=200,
        deterministic=False, # TODO: try True
        propagation_method=None,
        learn_logvar_bounds=False,
        activation_fn_cfg=None,
    )

    dynamics_model = models.OneDTransitionRewardModel(
        model,
        target_is_delta=True,
        normalize=True,
        normalize_double_precision=True,
        learned_rewards=True,
        obs_process_fn=None,
        no_delta_list=None,
        num_elites=5,
        model_normalization=model_normalization,
    )
    if model_dir:
        dynamics_model.load(model_dir)


def get_basic_buffer_iterators(
    replay_buffer,
    batch_size,
    val_ratio,
    ensemble_size=1,
    shuffle_each_epoch=True,
    bootstrap_permutes=False,
):
    data = replay_buffer.get_all(shuffle=True)
    val_size = int(replay_buffer.num_stored * val_ratio)
    train_size = replay_buffer.num_stored - val_size
    train_data = data[:train_size]
    train_iter = BootstrapIterator(
        train_data,
        batch_size,
        ensemble_size,
        shuffle_each_epoch=shuffle_each_epoch,
        permute_indices=bootstrap_permutes,
        rng=replay_buffer.rng,
    )

    val_iter = None
    if val_size > 0:
        val_data = data[train_size:]
        val_iter = TransitionIterator(
            val_data, batch_size, shuffle_each_epoch=False, rng=replay_buffer.rng
        )

    return train_iter, val_iter


def train_model_and_save_model_and_data(
    model,
    model_trainer,
    replay_buffer,
    work_dir,
    callback=None,
):
    dataset_train, dataset_val = get_basic_buffer_iterators(
        replay_buffer,
        256, # model batch size
        0.2, # validation ratio
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=False,
    )
    if hasattr(model, "update_normalizer"):
        model.update_normalizer(replay_buffer.get_all())
    # print(f"---------------model training-------------") # this part takes a lot of time
    training_losses, val_scores, l_infity_scores = model_trainer.train(
        dataset_train,
        dataset_val=dataset_val,
        num_epochs=None,
        patience=1,
        improvement_threshold=0.01,
        callback=callback,
    )
    if work_dir is not None:
        model.save(str(work_dir))
        replay_buffer.save(work_dir) # save the env data (real replay buffer)


def create_replay_buffer(
    datasize: int,
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    obs_type: Type = np.float32,
    action_type: Type = np.float32,
    reward_type: Type = np.float32,
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    collect_trajectories: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> ReplayBuffer:
    # print(f"dataset_size: {dataset_size}")
    dataset_size = datasize

    maybe_max_trajectory_len = None
    if collect_trajectories:
        maybe_max_trajectory_len = 64

    replay_buffer = ReplayBuffer(
        dataset_size,
        obs_shape,
        act_shape,
        obs_type=obs_type,
        action_type=action_type,
        reward_type=reward_type,
        rng=rng,
        max_trajectory_length=maybe_max_trajectory_len,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir))

    return 