import random
import collections
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Dict
from transformers import PreTrainedTokenizer

from textattack.attack_recipes import (
    PWWSRen2019,
    GeneticAlgorithmAlzantot2018,
    FasterGeneticAlgorithmJia2019,
    DeepWordBugGao2018,
    PSOZang2020,
    TextBuggerLi2018,
    BERTAttackLi2020,
    TextFoolerJin2019,
    HotFlipEbrahimi2017,
)
from textattack.models.wrappers import HuggingFaceModelWrapper, ModelWrapper
from textattack.datasets import HuggingFaceDataset

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.shared.attack import Attack
from textattack.transformations import (
    WordSwapEmbedding,
    WordSwapWordNet,
    CompositeTransformation,
    WordSwapMaskedLM,
)
from textattack.transformations import (
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)
from textattack.constraints.semantics import WordEmbeddingDistance

import textattack

from .instance import InputInstance


class CustomModelWrapper(HuggingFaceModelWrapper):

    def _model_predict(self, inputs):
        """Turn a list of dicts into a dict of lists.

        Then make lists (values of dict) into tensors.
        """
        input_dict = self.tokenizer(
            inputs,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        input_dict.to(model_device)

        outputs = self.model.predict(input_dict)

        return outputs["logits"]

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            # outputs = self._model_predict(inputs_dict)
            outputs = textattack.shared.utils.batch_model_predict(
                self._model_predict, text_input_list, batch_size=64
            )

        # for RSMI
        # outputs = textattack.shared.utils.batch_model_predict(
        #     self._model_predict, text_input_list, batch_size=64
        # )

        return outputs


class CustomTextAttackDataset(HuggingFaceDataset):
    """Loads a dataset from HuggingFace ``datasets`` and prepares it as a
    TextAttack dataset.

    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``datasets.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self,
        name,
        instances: List[InputInstance],
        label_map: Dict[str, int] = None,
        output_scale_factor=None,
        dataset_columns=None,
        shuffle=False,
    ):
        assert instances is not None or len(instances) == 0
        self._name = name
        self._i = 0
        self.label_map = label_map
        self.output_scale_factor = output_scale_factor
        self.label_names = sorted(list(label_map.keys()))

        if instances[0].is_nli():
            self.input_columns, self.output_column = ("premise", "hypothesis"), "label"
            self.examples = [
                {
                    "premise": instance.text_a,
                    "hypothesis": instance.text_b,
                    "label": int(instance.label),
                }
                for instance in instances
            ]
        else:
            self.input_columns, self.output_column = ("text",), "label"
            self.examples = [
                {"text": instance.text_a, "label": int(instance.label)}
                for instance in instances
            ]

        self.shuffled = shuffle
        if shuffle:
            random.shuffle(self.examples)

    @classmethod
    def from_instances(
        cls, name: str, instances: List[InputInstance], labels: Dict[str, int]
    ) -> "CustomTextAttackDataset":
        return cls(name, instances, labels)


def build_english_attacker(args, model) -> Attack:
    if args.attack_method == "hotflip":
        return HotFlipEbrahimi2017.build(model)
    if args.attack_method == "pwws":
        attacker = PWWSRen2019.build(model)
    elif args.attack_method == "pso":
        attacker = PSOZang2020.build(model)
    elif args.attack_method == "ga":
        attacker = GeneticAlgorithmAlzantot2018.build(model)
    elif args.attack_method == "fga":
        attacker = FasterGeneticAlgorithmJia2019.build(model)
    elif args.attack_method == "textfooler":
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == "bae":
        attacker = BERTAttackLi2020.build(model)
        attacker.transformation = WordSwapMaskedLM(
            method="bert-attack", max_candidates=args.neighbour_vocab_size
        )
    elif args.attack_method == "deepwordbug":
        attacker = DeepWordBugGao2018.build(model)
        attacker.transformation = CompositeTransformation(
            [
                # (1) Swap: Swap two adjacent letters in the word.
                WordSwapNeighboringCharacterSwap(),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                # (3) Deletion: Delete a random letter from the word.
                WordSwapRandomCharacterDeletion(),
                # (4) Insertion: Insert a random letter in the word.
                WordSwapRandomCharacterInsertion(),
            ],
            # total_count=args.neighbour_vocab_size
        )
    elif args.attack_method == "textbugger":
        attacker = TextBuggerLi2018.build(model)
    else:
        attacker = TextFoolerJin2019.build(model)

    if args.attack_method in ["textfooler", "pwws", "textbugger", "pso"]:
        attacker.transformation = WordSwapEmbedding(
            max_candidates=args.neighbour_vocab_size
        )
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
    attacker.constraints.append(MaxWordsPerturbed(max_percent=args.modify_ratio))
    use_constraint = UniversalSentenceEncoder(
        threshold=args.sentence_similarity,
        metric="angular",
        compare_against_original=False,
        window_size=15,
        skip_text_shorter_than_window=True,
    )
    attacker.constraints.append(use_constraint)

    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"}
    )
    attacker.pre_transformation_constraints.append(input_column_modification)

    attacker.goal_function = UntargetedClassification(
        model, query_budget=args.query_budget_size
    )
    return Attack(
        attacker.goal_function,
        attacker.constraints + attacker.pre_transformation_constraints,
        attacker.transformation,
        attacker.search_method,
    )
