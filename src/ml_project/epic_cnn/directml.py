"""DirectML-compatible loss and Adam updates for the EPIC CNN."""

import math

import torch


class StableBCEWithLogits(torch.autograd.Function):
    """Numerically stable unweighted BCE with an explicit sigmoid gradient."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(logits, targets)
        return (
            torch.maximum(logits, torch.zeros_like(logits))
            - logits * targets
            + torch.log1p(torch.exp(-torch.abs(logits)))
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, targets = ctx.saved_tensors
        grad_logits = torch.sigmoid(logits) - targets
        return grad_output * grad_logits, None


def dml_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return mean unweighted BCE while avoiding DirectML log-sigmoid fallback."""

    if logits.shape != targets.shape:
        raise ValueError("logits и targets должны иметь одинаковую форму")
    return StableBCEWithLogits.apply(logits, targets).mean()


def dml_weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Sum stable elementwise BCE using externally normalized loss weights."""

    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError("logits, targets и weights должны иметь одинаковую форму")
    losses = StableBCEWithLogits.apply(logits, targets)
    return (losses * weights).sum()


def dml_weighted_profile_bce_with_logits(
    plus_logits: torch.Tensor,
    minus_reverse_logits: torch.Tensor,
    plus_targets: torch.Tensor,
    minus_reverse_targets: torch.Tensor,
    plus_weights: torch.Tensor,
    minus_reverse_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted profile BCE with every alignment operation done on CPU."""

    if plus_logits.shape != minus_reverse_logits.shape:
        raise ValueError(
            "plus_logits и minus_reverse_logits должны иметь одинаковую форму"
        )
    if plus_logits.ndim != 3 or plus_logits.shape[1] != 1:
        raise ValueError("Каждый strand logit должен иметь форму [B,1,P]")
    tensors = (
        plus_targets,
        minus_reverse_targets,
        plus_weights,
        minus_reverse_weights,
    )
    if any(tensor.shape != plus_logits.shape for tensor in tensors):
        raise ValueError("Все targets и weights должны иметь форму [B,1,P]")
    plus_loss = StableBCEWithLogits.apply(plus_logits, plus_targets)
    minus_loss = StableBCEWithLogits.apply(
        minus_reverse_logits, minus_reverse_targets
    )
    return (
        (plus_loss * plus_weights).sum()
        + (minus_loss * minus_reverse_weights).sum()
    )


def dml_weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Weighted Huber sum expressed with DirectML-supported tensor operations."""

    if prediction.shape != target.shape or prediction.shape != weight.shape:
        raise ValueError("prediction, target и weight должны иметь одинаковую форму")
    if delta <= 0:
        raise ValueError("delta должен быть положительным")
    error = prediction - target
    absolute_error = torch.abs(error)
    quadratic = 0.5 * error.square()
    linear = delta * (absolute_error - 0.5 * delta)
    losses = torch.where(absolute_error <= delta, quadratic, linear)
    return (losses * weight).sum()


class DirectMLAdam(torch.optim.Optimizer):
    """Dense real-valued Adam expressed without the unsupported DML lerp op."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= betas[0] < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= betas[1] < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("DirectMLAdam requires dense gradients")
                if parameter.is_complex():
                    raise RuntimeError("DirectMLAdam requires real parameters")

                if group["weight_decay"] != 0:
                    gradient = gradient.add(
                        parameter,
                        alpha=group["weight_decay"],
                    )

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1)
                exp_avg.add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(
                    gradient,
                    gradient,
                    value=1 - beta2,
                )

                step = state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                denominator = exp_avg_sq.sqrt()
                denominator.div_(math.sqrt(bias_correction2))
                denominator.add_(group["eps"])

                step_size = group["lr"] / bias_correction1
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )

        return loss


__all__ = [
    "DirectMLAdam",
    "StableBCEWithLogits",
    "dml_bce_with_logits",
    "dml_weighted_bce_with_logits",
    "dml_weighted_huber",
    "dml_weighted_profile_bce_with_logits",
]
