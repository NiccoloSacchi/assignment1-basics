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
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Get state associated with p.
                state = self.state[p]
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
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

        self._init_args = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        defaults = self._init_args
        super().__init__(params, defaults)

    def init_args(self):
        return self._init_args

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

                state = self.state[p]  # Get state associated with p.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                # Initialize first and second moment vectors if this is the first time.
                if state.get("m") is None:
                    state["m"] = torch.zeros_like(grad)
                if state.get("v") is None:
                    state["v"] = torch.zeros_like(grad)

                # Update the first and second moment vectors.
                state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate. Step might be different from
                # iteration number as some parameters might be unused in forward pass or
                # have been added later to the model.
                t = state.get("step", 0)
                t += 1
                lr_corr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                state["step"] = t

                # Update parameters in place.
                p.data -= lr * weight_decay * p.data
                p.data -= lr_corr * state["m"] / (state["v"].sqrt() + eps)
        return loss


class CosineLearningRateScheduler:
    """
    CosineLearningRateScheduler applies a cosine annealing learning rate schedule
    with linear warmup to a PyTorch optimizer.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        max_learning_rate: The maximum learning rate after warmup.
        min_learning_rate: The minimum learning rate at the end of the cosine cycle.
        warmup_iters: Number of iterations for linear warmup.
        cosine_cycle_iters: Number of iterations for cosine annealing.

    Usage:
        scheduler = CosineLearningRateScheduler(optimizer, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        for iteration in range(num_iterations):
            lr = scheduler.step(iteration)  # Pass current iteration 0-indexed.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def step(self, iteration: int) -> float:
        # Update the learning rate in the optimizer.
        lr = learning_rate_schedule(
            it=iteration + 1,  # Convert to 1-indexed.
            max_learning_rate=self.max_learning_rate,
            min_learning_rate=self.min_learning_rate,
            warmup_iters=self.warmup_iters,
            cosine_cycle_iters=self.cosine_cycle_iters,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


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
        it (int): Iteration number to get learning rate for. 1-indexed.
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
        # return min_learning_rate + (max_learning_rate - min_learning_rate) * it / warmup_iters
        return max_learning_rate * it / warmup_iters

    if it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
            )
        ) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6

    # Collect all parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]

    if not params_with_grad:
        return

    # Compute the total L2 norm across all gradients
    total_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in params_with_grad))

    # If the total norm exceeds the maximum, scale all gradients
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)
