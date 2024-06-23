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


def evaluate(args):
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

    out_path = args.directory
    model.num_test_timesteps = 5

    model = load_model(model, os.path.join(out_path, "best_checkpoint.pt"), device)
    test_metric = evaluate_classify(
        model, dataset.get_datalaoder(split="test"), eval_logger
    )


def evaluate_diffuse(
    model,
    data_loader,
    logger,
):
    model.eval()
    metric = ClassificationMetric()
    with torch.inference_mode():
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

    #### for RSMI only
    # preds = []
    # labels = []
    # progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    # for (i, batch) in progress_bar:
    #     output = model.classify(batch)
    #     label = batch['label']
    #     logits = output['logits']
    #     loss = output['loss']
    #     metric(loss, logits, label)
    #     print(metric)
    # logger.info(metric)

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
        "--directory", "-d", type=str, required=True, help="Path to checkpoint."
    )

    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override options for `training` and/or `model` "
        "(format: parent.child=new_value format)",
    )

    args = parser.parse_args()

    evaluate(args)
