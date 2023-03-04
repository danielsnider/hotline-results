import datetime
import torch
import sys
from stable_diffusion_pytorch import pipeline

import hotline

prompts = ["a photograph of an astronaut riding a horse"]

# wait = 3
# warmup = 2
# active = 1
wait = 0
warmup = 0
active = 1
max_steps = wait + warmup + active

metadata= {
    'model': 'Stable Diffusion (Inference)',
    'dataset': 'LAION-5B',
    'batch_size': len(prompts),
    'optimizer': 'N/A',
    'runtime': [],
}

last_time = datetime.datetime.now()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active),
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('traces/'),
        on_trace_ready=hotline.analyze(
        None,
        None,
        run_name='Stable Diffusion (Inference)',
        test_accuracy=True,
        output_dir='/home/dans/cpath',
        metadata=metadata,
    ),
    record_shapes=False,
    profile_memory=False,
    with_stack=False
) as p:
  images = pipeline.generate(prompts, height=128, width=128, n_inference_steps=3)

  this_time = datetime.datetime.now()
  tdelta = this_time - last_time
  last_time = this_time
  print(f'step runtime: {tdelta}')
  metadata['runtime'].append(tdelta)




images[0].save('output.jpg')
