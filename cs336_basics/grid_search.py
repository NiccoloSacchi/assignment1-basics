"""Grid search script to train LLM models with different hyperparameter.

Example usage:
  .venv/bin/python cs336_basics/grid_search.py
"""

import subprocess

def run_training(args):
  """Run a single training job with given parameters."""
  cmd = [".venv/bin/python", "cs336_basics/train_llm.py"]
  
  # Add base arguments
  for key, value in args.items():
    if isinstance(value, list):
      cmd.extend([f"--{key}"] + [str(v) for v in value])
    else:
      cmd.extend([f"--{key}", str(value)])
  
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"Training failed for params {args}")
    print(f"Error: {result.stderr}")
    return False
  return True
    
def main():
  # Define base arguments. The model will be trained with these args and then 
  # overwritten by grid search parameters.
  base_args = {
    "train_tokens_path": "/Users/niccolosacchi/assignment1-basics/data/TinyStoriesV2-GPT4-train-tokens.npy",
    "val_tokens_path": "/Users/niccolosacchi/assignment1-basics/data/TinyStoriesV2-GPT4-valid-tokens.npy",
    "vocab_size": 10000,
    "context_length": 256,
    "num_layers": 4,
    "d_model": 512,
    "num_heads": 16,
    "d_ff": 1344,
    "rope_theta": 10000,
    "device": "mps",
    "betas": [0.9, 0.95],
    "weight_decay": 0.01,
    "warmup_iters": 100,
    "max_learning_rate": 1e-4,
    "min_learning_rate": 1e-5,
    "batch_size": 32,
    "pre_fetch_batches_mb": 8192,
    "total_tokens": 15_000_000, # 327_680_000
    "validation_interval": 100,
    "checkpoint_dir": "/Users/niccolosacchi/assignment1-basics/model/TinyStories",
    "checkpoint_interval": 1000,
    "wandb_project": "llm-simple-grid-search",
  }
  
  # Define grid search parameters. We will not try all combinations, but just
  # vary one grid_params per time to overwrite the base args.
  grid_params = {
    "context_length": [512],
    "d_model": [768],
    "batch_size": [16, 64],
    "num_layers": [8],
    "max_learning_rate": [1e-3],
    "weight_decay": [0.1],
  }

  # Count total runs.
  total_runs = 1
  for _, values in grid_params.items():
    total_runs += len(values)
  
  print(f"Starting grid search with {total_runs} total runs.")
  
  successful_runs = 0
  # Run with base args first.
  if run_training(base_args):
    successful_runs += 1
  for param_name, values in grid_params.items():
    for value in values:
      print(f"Running grid search with {param_name}={value}")
      args = base_args.copy()
      args[param_name] = value          
      if run_training(args):
        successful_runs += 1
      
  print(f"Grid search completed: {successful_runs}/{total_runs} runs successful")

if __name__ == "__main__":
  main()
