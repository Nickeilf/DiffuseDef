import argparse
import time
import os
import operator
from pathlib import Path
from functools import reduce
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import transformers
from torch.cuda.amp import autocast, GradScaler
from torch import nn

from diffdef.data.dataset import GroupedDataset
from diffdef.models import MODEL_CLASSES
from diffdef.utils.io import save_model, load_model
from diffdef.utils.logger import setup_logger
from diffdef.utils.metrics import ClassificationMetric
from diffdef.utils.misc import fix_seed, get_n_params
from diffdef.utils.optimizer import get_optimizer


def train(args):
    # Create a basic config with training defaults
    conf = OmegaConf.load(args.config)

    # Override with the CLI args
    cli_conf = OmegaConf.from_dotlist(args.overrides)

    # Merge config file and the CLI overrides
    conf = OmegaConf.merge(conf, cli_conf)

    # Set experiment folder name
    out_path = conf.experiment.output_dir
    out_path = Path(out_path)
    out_path = out_path / time.strftime("%m%d_%H:%M")
    out_path.mkdir(parents=True, exist_ok=True)

    # setup logging
    log_path = out_path / "trainer.log"
    train_logger = setup_logger("train", log_path)
    eval_logger = setup_logger("eval", log_path)

    train_logger.oom = 0
    train_logger.info(f"Experiment folder is: {str(out_path)!r}")

    # Set seed
    train_logger.info(f"Setting all seeds to {conf.experiment.seed!r}")
    fix_seed(conf.experiment.seed)

    # Print & save the full configuration
    print(OmegaConf.to_yaml(conf))
    OmegaConf.save(config=conf, f=str(out_path / "config.yaml"))

    ###################
    # Create the data #
    ###################
    n_epoch = conf.experiment.max_epoch
    device = conf.experiment.device
    pretrained_model = conf.model.pretrained_model

    dataset = GroupedDataset(
        conf.data,
        train_logger,
        tokenizer=pretrained_model,
        max_seq_len=conf.data.max_seq_len,
        device=device,
    )

    # build model
    model = MODEL_CLASSES[conf.model.type](args=conf.model)
    model.to(device)

    train_logger.info(model)
    train_logger.info(get_n_params(model))
    num_train_steps = dataset.get_train_steps() * n_epoch

    optimizer, scheduler = get_optimizer(
        conf.optimizer, model._get_learnable_params(), num_train_steps
    )

    best_metric = None
    best_checkpoint = 0
    for i in range(n_epoch):
        train_logger.info("Start training epoch {}".format(i + 1))
        train_epoch(
            model,
            dataset.get_datalaoder(split="train"),
            train_logger,
            optimizer,
            scheduler,
            conf,
        )
        eval_metric = evaluate_diffuse(
            model, dataset.get_datalaoder(split="dev"), eval_logger
        )

        # metrics based on loss
        if best_metric is None or eval_metric < best_metric:
            # save checkpoint
            best_metric = eval_metric
            save_model(model, optimizer, i, out_path, "best_checkpoint.pt")
            best_checkpoint = i + 1

    train_logger.info(
        f"Training complete. Best checkpoint achieved at epoch {best_checkpoint}"
    )
    eval_logger.info("Start testing model on the test set")
    model = load_model(model, os.path.join(out_path, "best_checkpoint.pt"), device)
    test_metric = evaluate_classify(
        model, dataset.get_datalaoder(split="test"), eval_logger
    )


def train_epoch(model, data_loader, logger, optimizer, scheduler, conf):
    amp_enabled = conf.experiment.enable_amp
    grad_scaler = GradScaler(enabled=amp_enabled)

    grad_clip = conf.optimizer.grad_clip

    model.train()

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, batch in progress_bar:
        try:
            with autocast(enabled=amp_enabled):
                output = model(batch)
                loss = output["loss"]
                progress_bar.set_description("loss: {:.4f}".format(loss.item()))

            grad_scaler.scale(loss).backward()

            # Apply gradient clipping if any
            if grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                params = reduce(
                    operator.add,
                    [group["params"] for group in optimizer.param_groups],
                )
                nn.utils.clip_grad_norm_(params, grad_clip)

            # Update params
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

            # lr scheduler step
            scheduler.step()

        except RuntimeError as e:
            if (
                "out of memory" in e.args[0]
                or "CUDA" in e.args[0]
                or "cuDNN" in e.args[0]
            ):
                torch.cuda.empty_cache()
                model.zero_grad()
                optimizer.zero_grad()
                logger.oom += 1
            else:
                raise e


def evaluate_diffuse(
    model,
    data_loader,
    logger,
):
    model.eval()
    metric = ClassificationMetric()
    # with torch.inference_mode():
    losses = []
    for i, batch in enumerate(data_loader):
        output = model(batch)
        loss = output["loss"].item()
        losses.append(loss)

    final_loss = sum(losses) / len(losses)
    logger.info("loss: {:.4f}".format(final_loss))

    return final_loss


def evaluate_classify(
    model,
    data_loader,
    logger,
):
    model.eval()
    metric = ClassificationMetric()
    with torch.inference_mode():
        preds = []
        labels = []
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, batch in progress_bar:
            output = model.classify(batch)
            label = batch["label"]
            logits = output["logits"]
            loss = output["loss"]
            metric(loss, logits, label)
            print(metric)

        logger.info(metric)
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Main training script",
    )

    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to configuration file."
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override options for `training` and/or `model` "
        "(format: parent.child=new_value format)",
    )

    args = parser.parse_args()

    train(args)
