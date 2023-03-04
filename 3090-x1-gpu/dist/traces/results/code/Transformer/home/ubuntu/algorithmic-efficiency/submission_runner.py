r"""Run a submission on a single workload.

# pylint: disable=line-too-long
Example command:

python3 submission_runner.py \
    --workload=mnist \
    --framework=jax \
    --submission_path=reference_algorithms/development_algorithms/mnist/mnist_jax/submission.py \
    --tuning_ruleset=external \
    --tuning_search_space=reference_algorithms/development_algorithms/mnist/tuning_search_space.json \
    --num_tuning_trials=3 \
    --experiment_dir=/home/username/codes/algorithmic-efficiency/experiment_dir

# pylint: enable=line-too-long
"""
import importlib
import inspect
import json
import os
import struct
import time
from typing import Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import torch
import torch.distributed as dist

from algorithmic_efficiency import halton
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.logger_utils import get_meta_data
from algorithmic_efficiency.logger_utils import set_up_loggers
from algorithmic_efficiency.profiler import PassThroughProfiler
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency.pytorch_utils import pytorch_init
from algorithmic_efficiency.pytorch_utils import pytorch_setup

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.set_visible_devices([], 'GPU')

# TODO(znado): make a nicer registry of workloads that lookup in.
BASE_WORKLOADS_DIR = 'algorithmic_efficiency/workloads/'

# Workload_path will be appended by '_pytorch' or '_jax' automatically.
WORKLOADS = {
    'cifar': {
        'workload_path': 'cifar/cifar', 'workload_class_name': 'CifarWorkload'
    },
    'criteo1tb': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallWorkload'
    },
    'fastmri': {
        'workload_path': 'fastmri/fastmri',
        'workload_class_name': 'FastMRIWorkload'
    },
    'imagenet_resnet': {
        'workload_path': 'imagenet_resnet/imagenet',
        'workload_class_name': 'ImagenetResNetWorkload'
    },
    'imagenet_vit': {
        'workload_path': 'imagenet_vit/imagenet',
        'workload_class_name': 'ImagenetVitWorkload'
    },
    'librispeech_conformer': {
        'workload_path': 'librispeech_conformer/librispeech',
        'workload_class_name': 'LibriSpeechConformerWorkload',
    },
    'librispeech_deepspeech': {
        'workload_path': 'librispeech_deepspeech/librispeech',
        'workload_class_name': 'LibriSpeechDeepSpeechWorkload',
    },
    'mnist': {
        'workload_path': 'mnist/mnist', 'workload_class_name': 'MnistWorkload'
    },
    'ogbg': {
        'workload_path': 'ogbg/ogbg', 'workload_class_name': 'OgbgWorkload'
    },
    'wmt': {'workload_path': 'wmt/wmt', 'workload_class_name': 'WmtWorkload'},
}

flags.DEFINE_string(
    'submission_path',
    None,
    'The relative path of the Python file containing submission functions. '
    'NOTE: the submission dir must have an __init__.py file!')
flags.DEFINE_string(
    'workload',
    None,
    help=f'The name of the workload to run.\n Choices: {list(WORKLOADS.keys())}'
)
flags.DEFINE_enum(
    'tuning_ruleset',
    'external',
    enum_values=['external', 'self'],
    help='Which tuning ruleset to use.')
flags.DEFINE_string(
    'tuning_search_space',
    None,
    'The path to the JSON file describing the external tuning search space.')
flags.DEFINE_integer('num_tuning_trials',
                     1,
                     'The number of external hyperparameter trials to run.')
flags.DEFINE_string('data_dir', '~/tensorflow_datasets/', 'Dataset location.')
flags.DEFINE_string('imagenet_v2_data_dir',
                    '~/tensorflow_datasets/',
                    'Dataset location for ImageNet-v2.')
flags.DEFINE_enum(
    'framework',
    None,
    enum_values=['jax', 'pytorch'],
    help='Whether to use Jax or Pytorch for the submission. Controls among '
    'other things if the Jax or Numpy RNG library is used for RNG.')
flags.DEFINE_string('tokenizer_vocab_path',
                    '',
                    'Location to read tokenizer from.')

flags.DEFINE_string(
    'experiment_dir',
    None,
    'The root directory to store all experiments. '
    'It is required and the directory should have '
    'an absolute path rather than a relative path.')
flags.DEFINE_string('experiment_name', None, 'Name of the experiment.')
flags.DEFINE_boolean('use_wandb',
                     False,
                     'Whether to use Weights & Biases logging.')
flags.DEFINE_boolean('profile', False, 'Whether to produce profiling output.')

FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, _ = pytorch_setup()


def convert_filepath_to_module(path: str):
  base, extension = os.path.splitext(path)

  if extension != '.py':
    raise ValueError(f'Path: {path} must be a python file (*.py)')

  return base.replace('/', '.')


def import_workload(workload_path: str,
                    workload_class_name: str,
                    return_class=False) -> spec.Workload:
  """Import and add the workload to the registry.

  This importlib loading is nice to have because it allows runners to avoid
  installing the dependencies of all the supported frameworks. For example, if
  a submitter only wants to write Jax code, the try/except below will catch
  the import errors caused if they do not have the PyTorch dependencies
  installed on their system.

  Args:
    workload_path: the path to the `workload.py` file to load.
    workload_class_name: the name of the Workload class that implements the
      `Workload` abstract class in `spec.py`.
    return_class: if true, then the workload class is returned instead of the
      instantiated object. Useful for testing when methods need to be overriden.
  """

  # Remove the trailing '.py' and convert the filepath to a Python module.
  workload_path = convert_filepath_to_module(workload_path)

  # Import the workload module.
  workload_module = importlib.import_module(workload_path)
  # Get everything defined in the workload module (including our class).
  workload_module_members = inspect.getmembers(workload_module)
  workload_class = None
  for name, value in workload_module_members:
    if name == workload_class_name:
      workload_class = value
      break
  if workload_class is None:
    raise ValueError(
        f'Could not find member {workload_class_name} in {workload_path}. '
        'Make sure the Workload class is spelled correctly and defined in '
        'the top scope of the module.')
  if return_class:
    return workload_class
  return workload_class()


def train_once(
    workload: spec.Workload,
    global_batch_size: int,
    data_dir: str,
    imagenet_v2_data_dir: str,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    hyperparameters: Optional[spec.Hyperparameters],
    rng: spec.RandomState,
    profiler: Profiler,
    log_dir: Optional[str] = None,
    tokenizer_vocab_path: Optional[str] = None
) -> Tuple[spec.Timing, spec.Steps]:
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)

  # Logger setup.
  logging.info('Initializing logger.')
  metrics_logger = None
  if log_dir is not None:
    hparams_filename = os.path.join(log_dir, 'hparams.json')
    meta_data = get_meta_data(workload)
    meta_filename = os.path.join(log_dir, 'meta_data.json')
    flag_filename = os.path.join(log_dir, 'flags.json')

    if RANK == 0:
      logging.info('Saving hparams to %s', hparams_filename)
      with open(hparams_filename, 'w') as f:
        f.write(json.dumps(hyperparameters._asdict(), indent=2))
      logging.info('Saving meta data to %s', meta_filename)
      with open(meta_filename, 'w') as f:
        f.write(json.dumps(meta_data, indent=2))
      logging.info('Saving flags to %s', flag_filename)
      with open(flag_filename, 'w') as f:
        f.write(json.dumps(flags.FLAGS.flag_values_dict(), indent=2))
      metrics_logger = set_up_loggers(log_dir, flags.FLAGS)

  # Workload setup.
  logging.info('Initializing dataset.')
  with profiler.profile('Initializing dataset'):
    input_queue = workload._build_input_queue(
        data_rng,
        'train',
        data_dir=data_dir,
        global_batch_size=global_batch_size)
  logging.info('Initializing model.')
  with profiler.profile('Initializing model'):
    model_params, model_state = workload.init_model_fn(model_init_rng)
  logging.info('Initializing optimizer.')
  # from IPython import embed
  # embed()

  with profiler.profile('Initializing optimizer'):
    optimizer_state = init_optimizer_state(workload,
                                           model_params,
                                           model_state,
                                           hyperparameters,
                                           opt_init_rng)

  logging.info('Initializing metrics bundle.')
  if tokenizer_vocab_path:
    workload.init_tokenizer(tokenizer_vocab_path)

  # Bookkeeping.
  goal_reached = False
  is_time_remaining = True
  last_eval_time = 0
  accumulated_submission_time = 0
  eval_results = []
  global_step = 0
  training_complete = False
  global_start_time = time.time()


  # Hotline profiling
  import hotline
  from IPython import embed
  import datetime
  """ Usage:
sudo apt-get install python3-venv
python3 -m venv env
source env/bin/activate

pip3 install -e '.[jax_cpu]'
pip3 install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'
pip3 install -e '.[full]'

pip install -e /home/dans/cpath

export CUDA_VISIBLE_DEVICES=0,1

source env/bin/activate
kill %; python3 submission_runner.py \
    --framework=pytorch \
    --workload=wmt \
    --experiment_dir=/home/dans/algorithmic-efficiency/experiment_dir \
    --submission_path=reference_algorithms/development_algorithms/wmt/wmt_pytorch/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/wmt/tuning_search_space.json \
    --data_dir=/FS1/dataset/wmt17_translate
  """

  last_time = datetime.datetime.now()
  print(last_time)

  quick_run = os.environ.get('HOTLINE_QUICK_RUN')
  if quick_run:
      wait = 1
      warmup = 0
      active = 1
  else:
      wait = 20
      warmup = 19
      active = 1
      # wait = 3
      # warmup = 3
      # active = 1
  max_steps = wait + warmup + active

  metadata = {
    'model': 'Transformer',
    'dataset': 'WMT17',
    'batch_size': global_batch_size,
    'optimizer': 'Adam',
    'runtime': [],
  }

  model_params = torch.nn.DataParallel(model_params)
  torch_profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active),
    on_trace_ready=hotline.analyze(
        model_params,
        input_queue,
        run_name='Transformer',
        test_accuracy=True,
        output_dir='/home/dans/cpath',
        metadata=metadata,
    ),
    record_shapes=False,
    profile_memory=False,
    with_stack=False
  )


  logging.info('Starting training loop.')
  while is_time_remaining and not goal_reached and not training_complete:
    step_rng = prng.fold_in(rng, global_step)
    data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)
    start_time = time.time()

    with profiler.profile('Data selection'):
      with hotline.annotate('Load Data'):
        batch = data_selection(workload,
                              input_queue,
                              optimizer_state,
                              model_params,
                              model_state,
                              hyperparameters,
                              global_step,
                              data_select_rng)
    try:
      with profiler.profile('Update parameters'):
        optimizer_state, model_params, model_state = update_params(
            workload=workload,
            current_param_container=model_params,
            current_params_types=workload.model_params_types,
            model_state=model_state,
            hyperparameters=hyperparameters,
            batch=batch,
            loss_type=workload.loss_type,
            optimizer_state=optimizer_state,
            eval_results=eval_results,
            global_step=global_step,
            rng=update_rng)

      this_time = datetime.datetime.now()
      tdelta = this_time - last_time
      logging.info(f'tdelta: {tdelta}')
      metadata['runtime'].append(tdelta)
      last_time = this_time
      logging.info(f'global_step: {global_step}\n')
      torch_profiler.step()
      hotline.annotate.step()
      if global_step >= max_steps:
        import sys
        sys.exit(0)
      global_step += 1
      continue

    except spec.TrainingCompleteError:
      training_complete = True
    global_step += 1
    if USE_PYTORCH_DDP:
      # Make sure all processes run eval after the same step when using DDP.
      dist.barrier()
    current_time = time.time()
    accumulated_submission_time += current_time - start_time
    is_time_remaining = (
        accumulated_submission_time < workload.max_allowed_runtime_sec)
    # Check if submission is eligible for an untimed eval.
    if (current_time - last_eval_time >= workload.eval_period_time_sec or
        training_complete):
      with profiler.profile('Evaluation'):
        try:
          latest_eval_result = workload.eval_model(global_batch_size,
                                                   model_params,
                                                   model_state,
                                                   eval_rng,
                                                   data_dir,
                                                   imagenet_v2_data_dir,
                                                   global_step)
          logging.info('%.2fs \t%d \t%s',
                       current_time - global_start_time,
                       global_step,
                       latest_eval_result)
          last_eval_time = current_time
          eval_results.append((global_step, latest_eval_result))

          if RANK == 0 and metrics_logger is not None:
            metrics_logger.append_scalar_metrics(
                latest_eval_result, global_step=global_step)

          goal_reached = workload.has_reached_goal(latest_eval_result)
        except RuntimeError as e:
          logging.exception(f'Eval step {global_step} error.\n')
          if 'out of memory' in str(e):
            logging.warning(
                f'error: GPU out of memory during eval during step {global_step}, error : {str(e)}'  # pylint: disable=line-too-long
            )
            if torch.cuda.is_available():
              torch.cuda.empty_cache()

  metrics = {'eval_results': eval_results, 'global_step': global_step}
  if USE_PYTORCH_DDP:
    # Sync final score (accumulated training time); choose highest, i.e. worst.
    dist.barrier()
    score_tensor = torch.tensor(accumulated_submission_time, device=DEVICE)
    dist.all_reduce(score_tensor, op=dist.ReduceOp.MAX)
    accumulated_submission_time = score_tensor.item()

  if RANK == 0 and metrics_logger is not None:
    metrics_logger.append_scalar_metrics({'score': accumulated_submission_time},
                                         global_step=global_step)
    metrics_logger.finish()

  return accumulated_submission_time, metrics


def score_submission_on_workload(workload: spec.Workload,
                                 workload_name: str,
                                 submission_path: str,
                                 data_dir: str,
                                 imagenet_v2_data_dir: str,
                                 profiler: Profiler,
                                 tuning_ruleset: str,
                                 tuning_search_space: Optional[str] = None,
                                 num_tuning_trials: Optional[int] = None,
                                 log_dir: Optional[str] = None,
                                 tokenizer_vocab_path: Optional[str] = None):
  # Expand paths because '~' may not be recognized
  data_dir = os.path.expanduser(data_dir)
  imagenet_v2_data_dir = os.path.expanduser(imagenet_v2_data_dir)

  # Remove the trailing '.py' and convert the filepath to a Python module.
  submission_module_path = convert_filepath_to_module(submission_path)
  submission_module = importlib.import_module(submission_module_path)

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  global_batch_size = submission_module.get_batch_size(workload_name)

  if tuning_ruleset == 'external':
    # If the submission runner is responsible for hyperparameter tuning, load in
    # the search space and generate a list of randomly selected hyperparameter
    # settings from it.
    if tuning_search_space is None:
      raise ValueError(
          'Must provide a tuning search space JSON file when using external '
          'tuning.')
    with open(tuning_search_space, 'r', encoding='UTF-8') as search_space_file:
      tuning_search_space = halton.generate_search(
          json.load(search_space_file), num_tuning_trials)
    all_timings = []
    all_metrics = []
    for hi, hyperparameters in enumerate(tuning_search_space):
      # Generate a new seed from hardware sources of randomness for each trial.
      rng_seed = struct.unpack('I', os.urandom(4))[0]
      rng = prng.PRNGKey(rng_seed)
      # Because we initialize the PRNGKey with only a single 32 bit int, in the
      # Jax implementation this means that rng[0] is all zeros, which means this
      # could lead to unintentionally reusing the same seed of only rng[0] were
      # ever used. By splitting the rng into 2, we mix the lower and upper 32
      # bit ints, ensuring we can safely use either rng[0] or rng[1] as a random
      # number.
      rng, _ = prng.split(rng, 2)
      logging.info('--- Tuning run %d/%d ---', hi + 1, num_tuning_trials)

      tuning_log_dir = None
      if log_dir is not None:
        tuning_log_dir = os.path.join(log_dir, str(hi + 1))
        if RANK == 0:
          logging.info('Creating tuning directory at %s', tuning_log_dir)
          os.makedirs(tuning_log_dir, exist_ok=True)

      with profiler.profile('Train'):
        if 'imagenet' not in workload_name:
          imagenet_v2_data_dir = None
        timing, metrics = train_once(workload, global_batch_size,
                                     data_dir, imagenet_v2_data_dir,
                                     init_optimizer_state,
                                     update_params, data_selection,
                                     hyperparameters, rng, profiler,
                                     tuning_log_dir,
                                     tokenizer_vocab_path)
      all_timings.append(timing)
      all_metrics.append(metrics)
    score = min(all_timings)
    for ti in range(num_tuning_trials):
      logging.info('Tuning trial %d/%d', ti + 1, num_tuning_trials)
      logging.info('Hyperparameters: %s', tuning_search_space[ti])
      logging.info('Metrics: %s', all_metrics[ti])
      logging.info('Timing: %s', all_timings[ti])
      logging.info('=' * 20)
  else:
    rng_seed = struct.unpack('q', os.urandom(8))[0]
    rng = prng.PRNGKey(rng_seed)
    # If the submission is responsible for tuning itself, we only need to run it
    # once and return the total time.
    with profiler.profile('Train'):
      score, _ = train_once(
          workload, global_batch_size, data_dir,
          imagenet_v2_data_dir,
          init_optimizer_state, update_params, data_selection,
          None, rng, profiler, log_dir, tokenizer_vocab_path)
  # TODO(znado): record and return other information (number of steps).
  return score


def main(_):
  if FLAGS.profile:
    profiler = Profiler()
  else:
    profiler = PassThroughProfiler()

  if FLAGS.framework == 'pytorch':
    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

  workload_metadata = WORKLOADS[FLAGS.workload]
  # Extend path according to framework.
  workload_metadata['workload_path'] = os.path.join(
      BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + '_' + FLAGS.framework,
      'workload.py')
  workload = import_workload(
      workload_path=workload_metadata['workload_path'],
      workload_class_name=workload_metadata['workload_class_name'])

  workload_dir_name = FLAGS.workload + '_' + FLAGS.framework
  if FLAGS.experiment_name is None:
    experiment_log_dir = os.path.join(FLAGS.experiment_dir, workload_dir_name)
  else:
    experiment_log_dir = os.path.join(FLAGS.experiment_dir,
                                      FLAGS.experiment_name,
                                      workload_dir_name)
  experiment_log_dir = os.path.expanduser(experiment_log_dir)

  if RANK == 0:
    # Only one worker should create the required dir.
    logging.info('Creating experiment directory at %s', experiment_log_dir)
    os.makedirs(name=experiment_log_dir, exist_ok=True)

  score = score_submission_on_workload(workload,
                                       FLAGS.workload,
                                       FLAGS.submission_path,
                                       FLAGS.data_dir,
                                       FLAGS.imagenet_v2_data_dir,
                                       profiler,
                                       FLAGS.tuning_ruleset,
                                       FLAGS.tuning_search_space,
                                       FLAGS.num_tuning_trials,
                                       experiment_log_dir,
                                       FLAGS.tokenizer_vocab_path)
  logging.info('Final %s score: %f', FLAGS.workload, score)

  if FLAGS.profile:
    logging.info(profiler.summary())

  if USE_PYTORCH_DDP:
    # Cleanup.
    dist.destroy_process_group()


if __name__ == '__main__':
  flags.mark_flag_as_required('workload')
  flags.mark_flag_as_required('framework')
  flags.mark_flag_as_required('submission_path')
  flags.mark_flag_as_required('experiment_dir')
  app.run(main)
