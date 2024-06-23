import torch
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from ..utils.io import read_cls_data


class Collater:
    def __init__(self, tokenizer, max_seq_len=256, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        output = {}

        idxs = [example["idx"] for example in batch]
        texts = [example["text_a"] for example in batch]
        labels = [example["label"] for example in batch]

        text_bs = [example["text_b"] for example in batch]
        if len(text_bs) == 0 or not text_bs[0]:
            text_bs = None

        output["idx"] = idxs
        output["label"] = torch.LongTensor(labels)

        # limit training time max sequence length to ensure fair comparison
        # this won't affect inference time (adv attack), which should tokenize the whole text
        tokenized_outputs = self.tokenizer(
            texts,
            text_bs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )
        output["text"] = tokenized_outputs

        for key, value in output.items():
            if key != "idx":
                output[key] = value.to(self.device)

        return output


class ClassificationDataset(Dataset):
    def __init__(self, fname):
        super().__init__()
        self.instances, self.label_map = read_cls_data(fname)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class GroupedDataset(object):
    def __init__(
        self,
        args,
        logger,
        tokenizer="bert-base-uncased",
        max_seq_len=256,
        device="cuda",
    ):
        self.logger = logger
        self.data_dir = Path(args.data_dir)
        self.train_file = self.data_dir / args.train
        self.dev_file = self.data_dir / args.dev
        self.test_file = self.data_dir / args.test
        self.num_label = args.num_label
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.collater = Collater(self.tokenizer, max_seq_len, self.device)

        # create data splits/dataloaders
        self.splits = self._create_splits()
        self.dataloaders = self._create_dataloader()

    def _create_splits(self):
        self.logger.info("Building train/dev/test splits ...")
        splits = {}
        splits["train"] = ClassificationDataset(self.train_file)
        splits["dev"] = ClassificationDataset(self.dev_file)
        splits["test"] = ClassificationDataset(self.test_file)

        for split in ["train", "dev", "test"]:
            self.logger.info(f"Number of {split} examples: {len(splits[split])}")

        return splits

    def _create_dataloader(self):
        self.logger.info("Building train/dev/test dataloaders ...")
        data_loaders = {}
        data_loaders["train"] = DataLoader(
            self.splits["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collater,
        )
        data_loaders["dev"] = DataLoader(
            self.splits["dev"],
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collater,
        )
        data_loaders["test"] = DataLoader(
            self.splits["test"],
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collater,
        )

        for split in ["train", "dev", "test"]:
            self.logger.info(f"Number of {split} steps: {len(data_loaders[split])}")

        return data_loaders

    def get_dataset(self, split="train"):
        return self.splits[split]

    def get_datalaoder(self, split="train"):
        return self.dataloaders[split]

    def get_train_steps(self):
        return len(self.dataloaders["train"])
