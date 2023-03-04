"""Training algorithm track submission functions for LibriSpeech."""
from typing import Dict, Iterator, List, Tuple
import os

import torch

from algorithmic_efficiency import spec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none")


import hotline


def get_batch_size(workload_name):
  # Return the global batch size.
  default = 256
  batch_sizes = {'librispeech': default}

  quick_run = os.environ.get('HOTLINE_QUICK_RUN')
  if quick_run:
    batch_sizes = {'librispeech': 8}  # smallest possible quick test
  else:
    gpu_model = torch.cuda.get_device_name(0)
    num_gpus = torch.cuda.device_count()
    if 'V100-SXM2-16GB' in gpu_model:
      batch_sizes = {'librispeech': min(32 * num_gpus, default)}
    elif '3090' in gpu_model:
      batch_sizes = {'librispeech': min(64  * num_gpus, default)}
    elif '2080' in gpu_model:
      batch_sizes = {'librispeech': min(8  * num_gpus, default)}

  from absl import logging
  logging.info(f'\n\nbatch_sizes: {batch_sizes}\n')

  return batch_sizes[workload_name]

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del workload
  del model_state
  del rng

  optimizer = torch.optim.Adam(model_params.parameters(),
                               hyperparameters.learning_rate)
  return optimizer


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    batch: Dict[str, spec.Tensor],
    # This will define the output activation via `output_activation_fn`.
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del current_params_types
  del eval_results
  del global_step
  del model_state
  del loss_type
  del hyperparameters

  with hotline.annotate('Forward'):
    (log_y, output_lengths), _ = workload.model_fn(
        current_param_container, batch, None,
        spec.ForwardPassMode.TRAIN, rng, False)

  with hotline.annotate('Calc Loss'):
    train_ctc_loss = torch.mean(workload.loss_fn(batch, (log_y, output_lengths)))

  with hotline.annotate('Zero Grad'):
    optimizer_state.zero_grad()

  with hotline.annotate('Backward'):
    train_ctc_loss.backward()

  with hotline.annotate('Optimizer'):
    optimizer_state.step()

  return optimizer_state, current_param_container, None


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.
  """
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  del hyperparameters
  del workload
  input_queue = iter(input_queue)
  return next(input_queue)
