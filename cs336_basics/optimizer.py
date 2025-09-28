from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    defaults = {"lr": lr}
    super().__init__(params, defaults)

  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      lr = group["lr"] # Get the learning rate.
      for p in group["params"]:
        if p.grad is None:
          continue
        state = self.state[p] # Get state associated with p.
        t = state.get("t", 0) # Get iteration number from the state, or initial value.
        grad = p.grad.data # Get the gradient of loss with respect to p.
        p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
    return loss


class AdamW(torch.optim.Optimizer):  
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    if eps < 0.0:
      raise ValueError(f"Invalid epsilon value: {eps}")
    if weight_decay < 0:
      raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    defaults = {
      "lr": lr,
      "betas": betas,
      "eps": eps,
      "weight_decay": weight_decay,
    }
    super().__init__(params, defaults)
  
  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
    
    for group in self.param_groups:
      lr = group["lr"]
      beta1, beta2 = group["betas"]
      eps = group["eps"]
      weight_decay = group["weight_decay"]

      for p in group["params"]:
        if p.grad is None:
          continue

        state = self.state[p] # Get state associated with p.
        grad = p.grad.data # Get the gradient of loss with respect to p.

        # Initialize first and second moment vectors if this is the first time.
        if state.get("m") is None:
          state["m"] = torch.zeros_like(grad)
        if state.get("v") is None:
          state["v"] = torch.zeros_like(grad)
        
        # Update the first and second moment vectors.
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * grad.pow(2)
        
        # Compute bias-corrected learning rate.
        t = state.get("t", 1)
        lr_corr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        state["t"] = t + 1

        # Update parameters in place.
        p.data -= lr * weight_decay * p.data
        p.data -= lr_corr * state["m"] / (state["v"].sqrt() + eps)
    return loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) * it / warmup_iters
    elif it < warmup_iters + cosine_cycle_iters:
        return (
          min_learning_rate +
          0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * (it - warmup_iters) / cosine_cycle_iters)
          )
        )
    return min_learning_rate
