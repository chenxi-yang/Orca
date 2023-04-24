# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import vrl.util.math
import vrl.modules.vrlnn as vrlnn

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init


class DeterministicMLP(Ensemble):
    """Implements a deterministic multi-layer perceptrons.

    It predicts per output mean and log variance(=0), and its weights are updated using a Gaussian
    negative log likelihood loss. 

    This class can also be used to build an ensemble of DeterministicMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size``.

    For the ensemble variant, uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    #### Not for DterministicMLP
    Valid propagation options are:
            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
        learn_logvar_bounds (bool): if ``True``, the logvar bounds will be learned, otherwise
            they will be constant. Defaults to ``False``.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ReLU when ``None``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic,
        )

        self.in_size = in_size
        self.out_size = out_size
        
        def create_activation():
            if activation_fn_cfg is None: # I would like to use relu
                activation_func = vrlnn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out):
            # return EnsembleLinearLayer(ensemble_size, l_in, l_out)
            return vrlnn.Linear(l_in, l_out)
            # return nn.Linear(l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), create_activation())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.mean_and_logvar = create_linear_layer(hid_size, out_size)
        # print(f"hid size:{hid_size}, out_size: {out_size}")

        self.apply(truncated_normal_init)
        self.to(self.device)
        self.elite_models: List[int] = None

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # self._maybe_toggle_layers_use_only_elite(only_elite)
        # print(f"Dynamics input: {self.in_size}; output: {self.out_size}")
        # if not isinstance(x, torch.Tensor):
            # print(f"input c: {x.c.shape}, width: {x.delta.shape}")
        # print(f"hidden layer weight: {self.hidden_layers['vrlnn.Linear']}")
        # if not isinstance(x, torch.Tensor):
        #     print(f"-- Dynamics Model --")
        #     print(f"Before hidden layer c: {x.c.cpu().detach().numpy().tolist()} width: {x.delta.cpu().detach().numpy().tolist()}")
        # if not isinstance(x, torch.Tensor):
        #     print(f"Symbolic Before hidden_layer: {x.c[0], x.delta[0]}")
        # else:
        #     print(f"Before hidden_layer: {x[0]}")
        x = self.hidden_layers(x)
        # if not isinstance(x, torch.Tensor):
        #     print(f"Symbolic Before mean: {x.c[0], x.delta[0]}")
        # else:
        #     print(f"Before mean: {x[0]}")
        mean_and_logvar = self.mean_and_logvar(x)
        # if not isinstance(x, torch.Tensor):
        #     print(f"Output c: {mean_and_logvar.c.cpu().detach().numpy().tolist()} width: {mean_and_logvar.delta.cpu().detach().numpy().tolist()}")
        #     print(f"-- End Dynamics Model --")
        # print(f"x: {x.shape}; last layer weight: {self.mean_and_logvar.weight.shape}")
        # self._maybe_toggle_layers_use_only_elite(only_elite)
        # if not isinstance(mean_and_logvar, torch.Tensor):
        #     print(f"Symbolic After mean: {mean_and_logvar.c[0], mean_and_logvar.delta[0]}")
        # else:
        #     print(f"After mean: {mean_and_logvar[0]}")

        return mean_and_logvar # , None # , None # forward the only batch
        # return mean_and_logvar[0], None# forward the only batch

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert(model_in.ndim == target.ndim)
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        # pred_mean, _ = self.forward(model_in, use_propagation=False)
        pred_mean = self.forward(model_in, use_propagation=False)
        # return F.mse_loss(pred_mean, target, reduction="none").sum()
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # print(f"model_in: {model_in.shape}")
        return self._mse_loss(model_in, target), {}

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert(model_in.ndim == 2 and target.ndim == 2)
        with torch.no_grad():
            # pred_mean, _ = self.forward(model_in, use_propagation=False)
            pred_mean = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1)) # no need for single model
            if pred_mean.ndim == 2:
                pred_mean = pred_mean[None, :]
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]
