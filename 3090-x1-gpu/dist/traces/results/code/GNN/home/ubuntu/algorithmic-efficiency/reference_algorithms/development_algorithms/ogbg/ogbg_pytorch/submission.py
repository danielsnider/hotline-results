from typing import Dict, Iterator, List, Tuple

import torch
import os

from algorithmic_efficiency import spec

import hotline

def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'ogbg': 32768}  # default
  batch_sizes = {'ogbg': 2048}  # largest possible on 2080

  quick_run = os.environ.get('HOTLINE_QUICK_RUN')
  if quick_run:
    batch_sizes = {'ogbg': 64}  # smallest possible quick test
  else:
    gpu_model = torch.cuda.get_device_name(0)
    num_gpus = torch.cuda.device_count()
    if 'V100-SXM2-16GB' in gpu_model:
      batch_sizes = {'ogbg': min(4096 * num_gpus, 32768)}
    elif '3090' in gpu_model:
      batch_sizes = {'ogbg': min(6144 * num_gpus, 32768)}

  from absl import logging
  logging.info(f'\n\nbatch_sizes: {batch_sizes}\n')
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an Adam optimizer."""
  del workload
  del model_state
  del rng
  optimizer_state = {
      'optimizer':
          torch.optim.Adam(
              model_params.parameters(), lr=hyperparameters.learning_rate)
  }
  return optimizer_state


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results
  del global_step

  current_model = current_param_container
  current_model.train()

  logits, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  with hotline.annotate('Calc Loss'):
    mask = batch['weights']
    loss, _ = workload.loss_fn(batch['targets'], logits, mask)

  with hotline.annotate('Zero Grad'):
    optimizer_state['optimizer'].zero_grad()

  with hotline.annotate('Backward'):
    loss.backward()

  with hotline.annotate('Optimizer'):
    optimizer_state['optimizer'].step()

  return optimizer_state, current_param_container, new_model_state


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  return next(input_queue)
