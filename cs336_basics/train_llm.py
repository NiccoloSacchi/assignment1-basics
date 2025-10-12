"""Train an LLM on the given dataset and with the specified configuration.

Example usage for TinyStories dataset:
  .venv/bin/python cs336_basics/train_llm.py \
    --train_tokens_path /Users/niccolosacchi/assignment1-basics/data/TinyStoriesV2-GPT4-train-tokens.npy \
    --val_tokens_path /Users/niccolosacchi/assignment1-basics/data/TinyStoriesV2-GPT4-valid-tokens.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --num_layers 4 \
    --d_model 512 \
    --num_heads 16 \
    --d_ff 1344 \
    --rope_theta 10000 \
    --device mps \
    --betas 0.9 0.95 \
    --weight_decay 0.1 \
    --warmup_iters 1000 \
    --max_learning_rate 1e-3 \
    --min_learning_rate 1e-5 \
    --batch_size 16 \
    --total_tokens 327680000 \
    --validation_interval 100 \
    --log_dir /Users/niccolosacchi/assignment1-basics/model/TinyStories/small_model \
    --checkpoint_dir /Users/niccolosacchi/assignment1-basics/model/TinyStories/small_model \
    --checkpoint_interval 100 \
    --load_checkpoint /Users/niccolosacchi/assignment1-basics/model/TinyStories/small_model/checkpoint_6.pt
  
Example usage for OpenWebText dataset:
  .venv/bin/python cs336_basics/train_llm.py \
    --train_tokens_path /Users/niccolosacchi/assignment1-basics/data/owt_train_tokens.bin \
    --train_metadata_path /Users/niccolosacchi/assignment1-basics/data/owt_train_tokens_metadata.json \
    --val_tokens_path /Users/niccolosacchi/assignment1-basics/data/owt_valid_tokens.bin \
    --val_metadata_path /Users/niccolosacchi/assignment1-basics/data/owt_valid_tokens_metadata.json \
    ...
"""

import argparse
import os
import torch
from cs336_basics.utils import (
  read_byte_file_to_memmap,
  get_batch,
  load_checkpoint,
  save_checkpoint,
)
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import (
  AdamW,
  CosineLearningRateScheduler,
  gradient_clipping,
)
from cs336_basics.loss import cross_entropy_loss
import time


# ============================================================================
# COMMAND LINE ARGUMENTS SETUP
# ============================================================================
parser = argparse.ArgumentParser(description="Train LLM script")

# Data.
parser.add_argument(
  '--train_tokens_path', type=str, required=True,
  help="Path to training tokens file.",
)
parser.add_argument(
  '--train_metadata_path', type=str, required=False,
  help="Path to training metadata. If passed, is used for reading the training tokens file.",
)
parser.add_argument(
  '--val_tokens_path', type=str, required=True,
  help="Path to validation tokens file.",
)
parser.add_argument(
  '--val_metadata_path', type=str, required=False,
  help="Path to validation metadata. If passed, is used for reading the validation tokens file.",
)

# Model hyperparameters.
parser.add_argument(
  '--vocab_size', type=int, required=True,
  help='Vocabulary size used to obtain the --train_tokens_path and --val_tokens_path files.',
)
parser.add_argument(
  '--context_length', type=int, required=True,
  help='Context length used for training.',
)
parser.add_argument(
  '--num_layers', type=int, required=True,
  help='Number of TransformerBlock layers.',
)
parser.add_argument(
  '--d_model', type=int, required=True,
  help='Hidden dimension of the model.',
)
parser.add_argument(
  '--num_heads', type=int, required=True,
  help='Number of attention heads.',
)
parser.add_argument(
  '--d_ff', type=int, required=True,
  help='Dimensionality of the position-wise feed-forward inner layer.',
)
parser.add_argument(
  '--rope_theta', type=float, required=True,
  help='If not None, use RoPE with the given base value to compute the rotation angles.',
)
parser.add_argument(
  '--device', type=str, default='cpu',
  help='PyTorch device string (e.g., cpu, cuda:0, or mps).',
)

# Optimizer hyperparameters
parser.add_argument(
  '--betas', type=float, nargs=2, required=True,
  help='Beta coefficients for AdamW.',
)  
parser.add_argument(
  '--weight_decay', type=float, required=True,
  help='Weight decay coefficient for AdamW.',
)

# Cosine learning rate scheduler hyperparameters.
parser.add_argument(
  '--warmup_iters', type=int, required=True,
  help='Number of iterations for linear warmup.',
)
parser.add_argument(
  '--max_learning_rate', type=float, required=True,
  help='Maximum learning rate after warmup.',
)
parser.add_argument(
  '--min_learning_rate', type=float, required=True,
  help='Minimum learning rate at the end of the cosine cycle.',
)

# Training hyperparameters.
parser.add_argument(
  '--batch_size', type=int, required=True,
  help='Batch size for training.',
)
parser.add_argument(
  '--total_tokens', type=int, required=True,
  help='Total number of tokens to process during training. total_iterations ~= total_tokens / (batch_size * context_length).',
)
parser.add_argument(
  '--validation_interval', type=int, required=True,
  help='Number of iterations between each validation.',
)

# Checkpointing.
parser.add_argument(
  '--checkpoint_dir', type=str, default=None,
  help='If passed, the model will be periodically saved at this location.',
)
parser.add_argument(
  '--checkpoint_interval', type=int, default=1000,
  help='Save a checkpoint every N iterations.',
)
parser.add_argument(
  '--load_checkpoint', type=str, default=None, required=False,
  help='Optional. Path to checkpoint file to resume training from.',
)

# Logging.
parser.add_argument(
  '--log_dir', type=str, default=None,
  help='If passed, training and validation losses will be logged to this directory.',
)

args = parser.parse_args()

# ============================================================================
# DATA LOADING
# ============================================================================
train_data = read_byte_file_to_memmap(
  args.train_tokens_path, args.train_metadata_path)
val_data = read_byte_file_to_memmap(
  args.val_tokens_path, args.val_metadata_path)

print("============================================================================")
print(f"Train data shape: {train_data.shape}, dtype: {train_data.dtype}")
print(f"Val data shape: {val_data.shape}, dtype: {val_data.dtype}")

# ============================================================================
# INSTANTIATE THE MODEL
# ============================================================================
model = TransformerLM(
  vocab_size=args.vocab_size,
  context_length=args.context_length,
  num_layers=args.num_layers,
  d_model=args.d_model,
  num_heads=args.num_heads,
  d_ff=args.d_ff,
  rope_theta=args.rope_theta,
  device=torch.device(args.device),
  dtype=torch.float32,
)
print("============================================================================")
print(f"Model instantiated on device {args.device}:")
print(model)

# ============================================================================
# INSTANTIATE THE OPTIMIZER AND LEARNING RATE SCHEDULER
# ============================================================================
optimizer = AdamW(
  model.parameters(),
  lr=args.min_learning_rate,  # Not really used, as we use a scheduler.
  betas=args.betas,
  eps=1e-8,
  weight_decay=args.weight_decay,
)
print("============================================================================")
print(f"Optimizer instantiated:")
print(optimizer)

total_iterations = args.total_tokens // (args.batch_size * args.context_length)
scheduler = CosineLearningRateScheduler(
  optimizer,
  max_learning_rate=args.max_learning_rate,
  min_learning_rate=args.min_learning_rate,
  warmup_iters=args.warmup_iters,
  cosine_cycle_iters=total_iterations,
)

# ============================================================================
# LOAD CHECKPOINT IF PROVIDED
# ============================================================================
start_iteration = 0
if args.load_checkpoint:
  print("============================================================================")
  print(f"Loading checkpoint from {args.load_checkpoint}")
  last_iteration = load_checkpoint(args.load_checkpoint, model, optimizer)
  start_iteration = last_iteration + 1
  print(f"Resuming training from iteration {start_iteration}")
  print("============================================================================")

# ============================================================================
# SETUP LOGGING AND CHECKPOINT DIRECTORIES
# ============================================================================
log_file = None
if args.log_dir:
  os.makedirs(args.log_dir, exist_ok=True)
  log_file = open(os.path.join(args.log_dir, "train.log"), "a")

if args.checkpoint_dir:
  os.makedirs(args.checkpoint_dir, exist_ok=True)
  
  # Save CLI command for loading checkpoints in a readme file.
  readme_path = os.path.join(args.checkpoint_dir, "README.md")
  cli_command = f"""# Model Checkpoints Directory
Resume training from a checkpoint in this directory with the following command:
```bash
.venv/bin/python cs336_basics/train_llm.py \\
--train_tokens_path {args.train_tokens_path} \\
--val_tokens_path {args.val_tokens_path} \\
--vocab_size {args.vocab_size} \\
--context_length {args.context_length} \\
--num_layers {args.num_layers} \\
--d_model {args.d_model} \\
--num_heads {args.num_heads} \\
--d_ff {args.d_ff} \\
--rope_theta {args.rope_theta} \\
--device {args.device} \\
--betas {args.betas[0]} {args.betas[1]} \\
--weight_decay {args.weight_decay} \\
--warmup_iters {args.warmup_iters} \\
--max_learning_rate {args.max_learning_rate} \\
--min_learning_rate {args.min_learning_rate} \\
--batch_size {args.batch_size} \\
--total_tokens {args.total_tokens} \\
--validation_interval {args.validation_interval} \\
--log_dir {args.log_dir} \\
--checkpoint_dir {args.checkpoint_dir} \\
--checkpoint_interval {args.checkpoint_interval} \\
--load_checkpoint <checkpoint_path>"""

  # Add optional arguments if they exist
  if args.train_metadata_path:
    cli_command += f" \\\n  --train_metadata_path {args.train_metadata_path}"
  if args.val_metadata_path:
    cli_command += f" \\\n  --val_metadata_path {args.val_metadata_path}"
  
  cli_command += "\n```\n"
  
  create_readme = True
  if os.path.exists(readme_path):
    print("============================================================================")
    create_readme = input(f"{readme_path} already exists. Overwrite? (y/n): ")
    if create_readme.lower() != 'y':
      print("Skipping README.md creation.")
      create_readme = False
  if create_readme:
    with open(readme_path, "w") as f:
      f.write(cli_command)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("============================================================================")

start_time = time.time()
step_width = len(str(total_iterations))
tokens_width = len(str(args.total_tokens))
for iteration in range(start_iteration, total_iterations):
  x, y = get_batch(
    dataset=train_data,
    batch_size=args.batch_size,
    context_length=args.context_length,
    device=args.device,
    dtype=torch.int32,
  )
  logits = model(x)
  train_loss = cross_entropy_loss(logits, y)
  optimizer.zero_grad()
  train_loss.backward()

  gradient_clipping(model.parameters(), max_l2_norm=1.0)
  lr = scheduler.step(iteration)
  optimizer.step()
  
  # Validation.
  current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
  elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
  loss_message = (
    f"[{current_time}] | elapsed={elapsed_str} | step={iteration:>{step_width}}/{total_iterations} | train_loss={train_loss.item():.4f}"
  )
  if iteration % args.validation_interval == 0:
    with torch.no_grad():
      val_x, val_y = get_batch(
        dataset=val_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
        dtype=torch.int32,
      )
      val_logits = model(val_x)
      val_loss = cross_entropy_loss(val_logits, val_y)
    loss_message += f" | val_loss={val_loss.item():.4f}"
  else:
    loss_message += " | val_loss=------"
  tokens_processed = iteration * args.batch_size * args.context_length
  print(f"{loss_message} | lr={lr:.6f} | tokens processed={tokens_processed:>{tokens_width}}/{args.total_tokens}")

  # Log training loss.
  if args.log_dir:
    log_file.write(loss_message + "\n")
    log_file.flush()

  # Save checkpoint.
  if args.checkpoint_dir and iteration != 0 and iteration % args.checkpoint_interval == 0:
      checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
      save_checkpoint(model, optimizer, iteration, checkpoint_path)
      print(f"Checkpoint saved at iteration {iteration}")

log_file.close()
