from typing import Dict, Iterator, List, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from algorithmic_efficiency import spec


import hotline
import os



def get_batch_size(workload_name):
  # Return the global batch size.
  default = 524_288
  batch_sizes = {'criteo1tb': default}
  batch_sizes = {'criteo1tb': 131_072}  # largest possible on 2080

  quick_run = os.environ.get('HOTLINE_QUICK_RUN')
  if quick_run:
    batch_sizes = {'criteo1tb': 2048}  # smallest possible quick test
  else:
    gpu_model = torch.cuda.get_device_name(0)
    num_gpus = torch.cuda.device_count()
    if 'V100-SXM2-16GB' in gpu_model:
      batch_sizes = {'criteo1tb': min(131_072 * num_gpus, default)}
    elif '3090' in gpu_model:
      batch_sizes = {'criteo1tb': min(196_608  * num_gpus, default)}
  from absl import logging
  logging.info(f'\n\num_gpus: {num_gpus}\n')
  logging.info(f'\n\nbatch_sizes: {batch_sizes}\n')

  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del rng
  del model_state

  base_lr = hyperparameters.learning_rate

  optimizer_state = {
      'optimizer':
          torch.optim.AdamW(
              model_params.parameters(),
              lr=base_lr,
              weight_decay=hyperparameters.weight_decay,
              betas=(hyperparameters.beta1, 0.999))
  }

  scheduler1 = LinearLR(
      optimizer_state['optimizer'],
      start_factor=1e-12,
      end_factor=1.,
      total_iters=hyperparameters.warmup_steps)

  scheduler2 = CosineAnnealingLR(
      optimizer_state['optimizer'],
      T_max=(workload.step_hint - hyperparameters.warmup_steps),
  )

  optimizer_state['scheduler'] = SequentialLR(
      optimizer_state['optimizer'],
      schedulers=[scheduler1, scheduler2],
      milestones=[hyperparameters.warmup_steps])

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
  """Return (updated_optimizer_state, updated_params)."""

  current_model = current_param_container
  current_param_container.train()

  with hotline.annotate('Forward'):
    logits_batch, new_model_state = workload.model_fn(
        params=current_model,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=False)

  with hotline.annotate('Calc Loss'):
    loss, _ = workload.loss_fn(
        label_batch=batch['targets'], logits_batch=logits_batch)

  with hotline.annotate('Zero Grad'):
    optimizer_state['optimizer'].zero_grad()

  with hotline.annotate('Backward'):
    loss.backward()

  with hotline.annotate('Optimizer'):
    optimizer_state['optimizer'].step()
    optimizer_state['scheduler'].step()

  return (optimizer_state, current_param_container, new_model_state)


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
