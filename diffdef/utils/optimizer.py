from typing import Iterator, Tuple
from omegaconf import DictConfig

from torch import optim, nn

from transformers import get_scheduler

KNOWN_SCHEDULERS = [
    "constant",
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "inverse_sqrt",
    "reduce_lr_on_plateau",
]


def get_optimizer(
    cfg: DictConfig,
    named_params: Iterator[Tuple[str, nn.Parameter]],
    num_train_steps: int,
):
    # Create parameter group, skip weight decay for layer norm and biases
    _skip = (".bias", ".LayerNorm.weight")

    cfg = cfg.copy()
    cfg.pop("grad_clip")

    # Pop args
    weight_decay = cfg.pop("weight_decay", 0.0)

    wd_params = [param for (name, param) in named_params if not name.endswith(_skip)]
    no_wd_params = [param for (name, param) in named_params if name.endswith(_skip)]
    param_groups = [
        {"params": wd_params, "weight_decay": weight_decay},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]

    # Create the actual optimizer
    sched_opts = cfg.pop("scheduler", {})
    optimizer = getattr(optim, cfg.pop("name"))(param_groups, **dict(cfg))

    sched_name = sched_opts.pop("name", "constant")

    # Handle warmup steps or ratio
    warmup_steps_or_ratio = sched_opts.pop("warmup_steps_or_ratio", 0)
    if isinstance(warmup_steps_or_ratio, float):
        num_warmup_steps = int(warmup_steps_or_ratio * num_train_steps)
    else:
        num_warmup_steps = warmup_steps_or_ratio

    assert sched_name in KNOWN_SCHEDULERS, f"Unknown LR scheduler {sched_name!r}"

    # Use 'constant' in config for both warmed up and the plain one
    if sched_name == "constant" and num_warmup_steps > 0:
        sched_name = "constant_with_warmup"

    scheduler = get_scheduler(
        sched_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    return optimizer, scheduler
