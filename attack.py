import argparse
import logging
import time
import os
import operator
import numpy as np
import torch
from pathlib import Path
from functools import reduce
from omegaconf import OmegaConf
from tqdm import tqdm

import textattack

# from textattack import Attacker
from textattack.loggers import AttackLogManager

from diffdef.data.dataset import GroupedDataset
from diffdef.models import MODEL_CLASSES
from diffdef.utils.io import save_model, load_model
from diffdef.utils.metrics import SimplifidResult
from diffdef.utils.misc import fix_seed
from diffdef.utils.attack_utils import (
    build_english_attacker,
    CustomTextAttackDataset,
    CustomModelWrapper,
)


def attack(args):
    # setup logging
    model_dir = Path(args.directory)
    log_path = model_dir / f"{args.attack_method}_attack_{args.denoise_step}_steps.txt"
    config = model_dir / "config.yaml"

    conf = OmegaConf.load(config)

    # Set seed
    train_logger = logging.getLogger(__name__)
    train_logger.setLevel(logging.DEBUG)
    train_logger.info(f"Setting all seeds to {42!r}")
    fix_seed(42)

    attacker_log_manager = AttackLogManager()
    attacker_log_manager.add_output_file(str(log_path))
    attacker_log_manager.enable_stdout()

    ###################
    # Create the data #
    ###################
    device = conf.experiment.device
    pretrained_model = conf.model.pretrained_model
    grouped_dataset = GroupedDataset(
        conf.data, train_logger, tokenizer=pretrained_model, device=device
    )

    test_instances = grouped_dataset.get_dataset("test").instances
    label_map = grouped_dataset.get_dataset("test").label_map

    # label_map = {k: str(v) for k, v in label_map.items()}

    test_instances = [x for x in test_instances if len(x.text_a.split(" ")) > 4]
    choice_instances = np.random.choice(
        test_instances, size=(args.attack_numbers,), replace=False
    )

    dataset = CustomTextAttackDataset.from_instances(
        "test", choice_instances, label_map
    )
    # choice_tuples = [x.get_tuple() for x in choice_instances]
    # dataset = textattack.datasets.TextAttackDataset(choice_tuples)

    # build model
    model = MODEL_CLASSES[conf.model.type](args=conf.model)
    model.to(device)
    model.eval()
    model = load_model(model, model_dir / "best_checkpoint.pt", device)

    # diffuse model denoise steps
    model.num_test_timesteps = args.denoise_step
    model.NUM_ENSEMBLE = args.num_ensemble

    print(f"Ensemble number {model.NUM_ENSEMBLE}")

    # build attacker
    with torch.inference_mode():
        model_wrapper = CustomModelWrapper(model, grouped_dataset.tokenizer)
        attacker = build_english_attacker(args, model_wrapper)

        # attack_args = textattack.AttackArgs(
        #     num_examples=-1,
        #     num_workers_per_device=30,
        # )
        # attacker = Attacker(attack, dataset, attack_args=attack_args)

        # attack
        results_iterable = attacker.attack_dataset(dataset)
        description = tqdm(results_iterable, total=len(choice_instances))
        result_statistics = SimplifidResult()

        for result in description:
            # try:
            attacker_log_manager.log_result(result)
            result_statistics(result)
            description.set_description(result_statistics.__str__())
            # except Exception as e:
            #     print('error in process')
            #     continue

        attacker_log_manager.log_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Main training script",
    )

    parser.add_argument(
        "--directory", "-d", type=str, required=True, help="Path to model folder."
    )

    parser.add_argument(
        "--attack-method",
        "-am",
        type=str,
        required=False,
        default="textfooler",
        help="Attack method.",
    )

    parser.add_argument(
        "--attack-numbers",
        "-an",
        type=int,
        required=False,
        default=1000,
        help="Number of examples to attack.",
    )

    parser.add_argument(
        "--query-budget-size",
        "-qbs",
        type=int,
        required=False,
        default=50,
        help="Number of query budget to attack.",
    )

    parser.add_argument(
        "--sentence_similarity",
        "-ss",
        type=float,
        required=False,
        default=0.840845057,
        help="USE sentence similarity threshold.",
    )

    parser.add_argument(
        "--modify-ratio",
        "-mr",
        type=float,
        required=False,
        default=0.3,
        help="Max ratio of words to modify.",
    )

    parser.add_argument(
        "--neighbour-vocab-size",
        "-nvs",
        type=int,
        required=False,
        default=50,
        help="Number of neighboring vocabulary to sample.",
    )

    parser.add_argument(
        "--denoise-step",
        "-ds",
        type=int,
        required=False,
        default=10,
        help="Number of denoising steps.",
    )

    parser.add_argument(
        "--num-ensemble",
        "-ne",
        type=int,
        required=False,
        default=10,
        help="Number of ensembing.",
    )

    args = parser.parse_args()
    attack(args)
