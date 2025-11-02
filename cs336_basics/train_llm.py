"""Train an LLM on the given dataset and with the specified configuration.

Example usages for TinyStories dataset:
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
    --weight_decay 0.01 \
    --warmup_iters 100 \
    --max_learning_rate 1e-3 \
    --min_learning_rate 1e-5 \
    --batch_size 32 \
    --pre_fetch_batches_mb 8192 \
    --total_tokens 327680000 \
    --validation_interval 100 \
    --checkpoint_dir /Users/niccolosacchi/assignment1-basics/model/TinyStories/ \
    --checkpoint_interval 1000
    
  .venv/bin/python cs336_basics/train_llm.py \
    --train_tokens_path /Users/niccolosacchi/assignment1-basics/data/TinyStoriesV2-GPT4-train-tokens.npy \
    --val_tokens_path /Users/niccolosacchi/assignment1-basics/data/TinyStoriesV2-GPT4-valid-tokens.npy \
    --device mps \
    --warmup_iters 1000 \
    --max_learning_rate 1e-3 \
    --min_learning_rate 1e-5 \
    --batch_size 16 \
    --total_tokens 327680000 \
    --validation_interval 100 \
    --checkpoint_dir /Users/niccolosacchi/assignment1-basics/model/TinyStories/small_model \
    --checkpoint_interval 100 \
    --load_checkpoint /Users/niccolosacchi/assignment1-basics/model/TinyStories/small_model/checkpoint_10.pt
  
Example usage for OpenWebText dataset:
  .venv/bin/python cs336_basics/train_llm.py \
    --train_tokens_path /Users/niccolosacchi/assignment1-basics/data/owt_train_tokens.bin \
    --train_metadata_path /Users/niccolosacchi/assignment1-basics/data/owt_train_tokens_metadata.json \
    --val_tokens_path /Users/niccolosacchi/assignment1-basics/data/owt_valid_tokens.bin \
    --val_metadata_path /Users/niccolosacchi/assignment1-basics/data/owt_valid_tokens_metadata.json \
    ...
"""

import wandb
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from cs336_basics.io import (
  read_byte_file_to_memmap,
  load_checkpoint,
  save_checkpoint,
  MemmapTokenDataset,
)
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import (
  AdamW,
  CosineLearningRateScheduler,
  gradient_clipping,
)
from cs336_basics.loss import cross_entropy_loss, perplexity
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
  '--vocab_size', type=int, required=False,
  help='Vocabulary size used to obtain the --train_tokens_path and --val_tokens_path files.',
)
parser.add_argument(
  '--context_length', type=int, required=False,
  help='Context length used for training.',
)
parser.add_argument(
  '--num_layers', type=int, required=False,
  help='Number of TransformerBlock layers.',
)
parser.add_argument(
  '--d_model', type=int, required=False,
  help='Hidden dimension of the model.',
)
parser.add_argument(
  '--num_heads', type=int, required=False,
  help='Number of attention heads.',
)
parser.add_argument(
  '--d_ff', type=int, required=False,
  help='Dimensionality of the position-wise feed-forward inner layer.',
)
parser.add_argument(
  '--rope_theta', type=float, required=False,
  help='If not None, use RoPE with the given base value to compute the rotation angles.',
)
parser.add_argument(
  '--device', type=str, default='cpu',
  help='PyTorch device string (e.g., cpu, cuda:0, or mps).',
)

# Optimizer hyperparameters
parser.add_argument(
  '--betas', type=float, nargs=2, required=False, default=(0.9, 0.95),
  help='Beta coefficients for AdamW.',
)  
parser.add_argument(
  '--weight_decay', type=float, required=False, default=0.0,
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
  '--pre_fetch_batches_mb', type=int, required=False, default=1024,
  help='How much training data to pre-fetch (in MB) for training.',
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
  help='Optional. Path to checkpoint file to resume training from. If passed, the above model and optimizer hyperparameters are ignored.',
)

parser.add_argument(
  '--wandb_project', type=str, default="llm-project", required=False,
  help='Optional. WandB project name for logging.',
)

args = parser.parse_args()

# ============================================================================
# LOAD CHECKPOINT IF PROVIDED
# ============================================================================
start_iteration = 0
model = None
optimizer = None
if args.load_checkpoint:
  print("============================================================================")
  print(f"Loading checkpoint from {args.load_checkpoint}")
  model, optimizer, last_iteration = load_checkpoint(
    args.load_checkpoint,
    TransformerLM,
    AdamW,
    device=args.device,
  )
  start_iteration = last_iteration + 1
  print(f"Resuming training from iteration {start_iteration}")
  print("============================================================================")
  print(f"Optimizer loaded from checkpoint and instantiated:")
  print(optimizer)
  print("============================================================================")
  print(f"Model loaded from checkpoint.")

# ============================================================================
# INSTANTIATE THE MODEL
# ============================================================================
if model is None:
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
device = next(model.parameters()).device
print(f"Model instantiated on device {device}:")
summary(
  model,
  input_data=torch.randint(0, args.vocab_size, (1, args.context_length)),
  col_names=["output_size", "num_params"],
  row_settings=["var_names"],
  depth=10,
)

# ============================================================================
# INSTANTIATE THE OPTIMIZER AND LEARNING RATE SCHEDULER
# ============================================================================
if optimizer is None:
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

context_length = model.init_args()["context_length"]
total_iterations = args.total_tokens // (args.batch_size * context_length)
scheduler = CosineLearningRateScheduler(
  optimizer,
  max_learning_rate=args.max_learning_rate,
  min_learning_rate=args.min_learning_rate,
  warmup_iters=args.warmup_iters,
  cosine_cycle_iters=total_iterations,
)
  
# ============================================================================
# DATA LOADING AND PREFETCHING SETUP
# ============================================================================
train_data = read_byte_file_to_memmap(
  args.train_tokens_path, args.train_metadata_path)
val_data = read_byte_file_to_memmap(
  args.val_tokens_path, args.val_metadata_path)

print("============================================================================")
print(f"Train data shape: {train_data.shape}, dtype: {train_data.dtype}")
print(f"Val data shape: {val_data.shape}, dtype: {val_data.dtype}")

batch_size_mb = args.batch_size * context_length * 4 // 1024  # Approximate batch size in MB (assuming int32 tokens).
prefetch_batches = args.pre_fetch_batches_mb // batch_size_mb
print(f"Pre-fetching {prefetch_batches} batches ({args.pre_fetch_batches_mb} MB) for training.")

train_batch_data = MemmapTokenDataset(
  memmap_data=train_data,
  batch_size=args.batch_size,
  context_length=context_length,
  device=args.device,
  dtype=torch.int32,
  prefetch_batches=prefetch_batches,
)
train_batch_data_loader = DataLoader(
  train_batch_data,
  batch_size=None,  # Batching handled by dataset
  num_workers=0,    # Keep 0 for memmap to avoid multiprocessing issues.
)
train_batch_data_iterator = iter(train_batch_data_loader)

val_batch_data = MemmapTokenDataset(
  memmap_data=val_data,
  batch_size=args.batch_size,
  context_length=context_length,
  device=args.device,
  dtype=torch.int32,
  prefetch_batches=10,  # Prefetch just a few batches for validation.
)
val_batch_data_loader = DataLoader(
  val_batch_data,
  batch_size=None,  # Batching handled by dataset.
  num_workers=0,    # Keep 0 for memmap to avoid multiprocessing issues.
)
val_batch_data_iterator = iter(val_batch_data_loader)

# ============================================================================
# Initialize Weights and Biases
# ============================================================================
# 1. Define your configuration (optional but recommended).
config = {
  "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
  "vocab_size": args.vocab_size,
  "context_length": args.context_length,
  "num_layers": args.num_layers,
  "d_model": args.d_model,
  "num_heads": args.num_heads,
  "d_ff": args.d_ff,
  "rope_theta": args.rope_theta,
  "device": args.device,
  "betas": args.betas,
  "weight_decay": args.weight_decay,
  "warmup_iters": args.warmup_iters,
  "max_learning_rate": args.max_learning_rate,
  "min_learning_rate": args.min_learning_rate,
  "batch_size": args.batch_size,
  "pre_fetch_batches_mb": args.pre_fetch_batches_mb,
  "total_tokens": args.total_tokens,
  "validation_interval": args.validation_interval,
  "checkpoint_dir": args.checkpoint_dir,
  "checkpoint_interval": args.checkpoint_interval,
}

# 2. Initialize a new run.
run = wandb.init(
  project=args.wandb_project,
  config=config,
)

# ============================================================================
# SETUP CHECKPOINT DIRECTORIES
# ============================================================================
checkpoint_dir = None
if args.checkpoint_dir:
  checkpoint_dir = args.checkpoint_dir + "/" + run.name
  os.makedirs(checkpoint_dir, exist_ok=True)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("============================================================================")

start_time = time.time()
step_width = len(str(total_iterations))
tokens_width = len(str(args.total_tokens))
try:
  for iteration in range(start_iteration, total_iterations):
    iteration_start_time = time.time()
    x, y = next(train_batch_data_iterator)
    logits = model(x)
    train_loss = cross_entropy_loss(logits, y)
    train_perplexity = perplexity(train_loss.item())
    optimizer.zero_grad()
    train_loss.backward()

    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    lr = scheduler.step(iteration)
    optimizer.step()
    
    # Validation.
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    total_duration_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    iteration_duration = time.time() - iteration_start_time
    iteration_duration_str = time.strftime("%H:%M:%S", time.gmtime(iteration_duration))
    loss_message = (
      f"[{current_time}] | total_duration={total_duration_str} | iteration_duration={iteration_duration_str} | step={iteration:>{step_width}}/{total_iterations} | train_loss={train_loss.item():.4f} | train_perplexity={train_perplexity:.4f}"
    )
    val_loss = None
    val_perplexity = None
    if iteration % args.validation_interval == 0:
      with torch.no_grad():
        val_x, val_y = next(val_batch_data_iterator)
        val_logits = model(val_x)
        val_loss = cross_entropy_loss(val_logits, val_y).item()
        val_perplexity = perplexity(val_loss)
      loss_message += f" | val_loss={val_loss:.4f} | val_perplexity={val_perplexity:.4f}"
    else:
      loss_message += " | val_loss=------ | val_perplexity=------"
    tokens_processed = iteration * args.batch_size * context_length
    loss_message = f"{loss_message} | lr={lr:.6f} | tokens processed={tokens_processed:>{tokens_width}}/{args.total_tokens}"
    print(loss_message)
    
    # Log to Weights and Biases.
    wandb.log(
      {
        "train/loss": train_loss.item(),
        "train/perplexity": train_perplexity,
        "val/loss": val_loss,
        "val/perplexity": val_perplexity,
        "learning_rate": lr,
        "tokens_processed": tokens_processed,
        "iteration_duration": iteration_duration,
      },
      step=iteration,
    )

    # Save checkpoint.
    if checkpoint_dir and iteration != 0 and iteration % args.checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
        save_checkpoint(model, optimizer, iteration, checkpoint_path)
        print(f"Checkpoint saved at iteration {iteration}")
except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
except Exception as e:
    print(f"\nTraining failed with error: {e}")
    raise
finally:
    wandb.finish()
    if checkpoint_dir and iteration != 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
        save_checkpoint(model, optimizer, iteration, checkpoint_path)
