# dataset.py
import csv
from pathlib import Path
from typing import Tuple, Dict, List

import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class MSMARCODataModule(pl.LightningDataModule):
    """Skeleton DataModule plug in your own Dataset implementation."""

    def __init__(self, tokenizer_name: str = "bert-base-uncased", batch_size: int = 32, num_workers: int = 8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):

        dataset_dir = "/work/hdd/bcsi/ndaithankar/datasets/msmarco/"
        self.train_set = StreamingMSMARCOTripleDataset(
            dataset_dir+"qidpidtriples.train.full.tsv",
            dataset_dir+"queries.train.tsv",
            dataset_dir+"collection.tsv",
            tokenizer=self.tokenizer,
            max_length=128,
        )
        # self.val_set = MSMARCOTripleDataset(
        #     dataset_dir+"triples.dev.small.tsv",  # or construct validation triples via qrels
        #     dataset_dir+"queries.dev.tsv",
        #     dataset_dir+"collection.tsv",
        #     tokenizer=self.tokenizer,
        #     max_length=128,
        # )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            # shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        # return torch.utils.data.DataLoader(
        #     self.val_set,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        # )
        return None


class MSMARCOTripleDataset(Dataset):
    """
    Triple-loader for MS MARCO Passage Ranking.
    Each item returns tokenised query / positive-passage / negative-passage.
    """

    def __init__(
        self,
        triples_path: str,
        queries_path: str,
        corpus_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        lowercase: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lowercase = lowercase

        # ---------- load queries -------------------------------------------------
        self.queries: Dict[str, str] = {}
        print(f"Loading queries from {queries_path} …")
        with open(queries_path, encoding="utf-8") as f:
            tsv = csv.reader(f, delimiter="\t")
            for qid, qtext in tsv:
                self.queries[qid] = qtext.lower() if lowercase else qtext

        # ---------- load corpus --------------------------------------------------
        # collection.tsv has 8.8M lines → read lazily into list for O(1) id lookup
        print(f"Loading corpus index from {corpus_path} … (this may take ~1-2 min)")
        self.passages: Dict[str, str] = {}
        with open(corpus_path, encoding="utf-8") as f:
            tsv = csv.reader(f, delimiter="\t")
            for pid, ptext in tsv:
                self.passages[pid] = ptext.lower() if lowercase else ptext

        # ---------- load triples -------------------------------------------------
        self.triples: List[Tuple[str, str, str]] = []
        print(f"Loading triples from {triples_path} …")
        with open(triples_path, encoding="utf-8") as f:
            tsv = csv.reader(f, delimiter="\t")
            for qid, pos_id, neg_id in tsv:
                qid, pos_id, neg_id = qid.strip(), pos_id.strip(), neg_id.strip()
                if qid in self.queries and pos_id in self.passages and neg_id in self.passages:
                    self.triples.append((qid, pos_id, neg_id))

        print(
            f"Loaded {len(self.triples):,} triples, "
            f"{len(self.queries):,} queries, "
            f"{len(self.passages):,} passages."
        )

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        qid, pos_id, neg_id = self.triples[idx]

        q_text   = self.queries[qid]
        pos_text = self.passages[pos_id]
        neg_text = self.passages[neg_id]

        q_enc   = self._encode(q_text)
        pos_enc = self._encode(pos_text)
        neg_enc = self._encode(neg_text)

        return {
            "query":   q_enc,
            "pos_doc": pos_enc,
            "neg_doc": neg_enc,
        }

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # squeeze to drop the batch dimension added by return_tensors="pt"
        return {k: v.squeeze(0) for k, v in enc.items()}
    

from torch.utils.data import IterableDataset
import csv

class StreamingMSMARCOTripleDataset(IterableDataset):
    """
    Streams large qidpid triple file without loading into memory.
    Each line: qid<TAB>pos_pid<TAB>neg_pid
    """

    def __init__(self, triples_path, queries_path, corpus_path, tokenizer, max_length=128):
        self.triples_path = triples_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load query and passage text dictionaries into memory
        self.queries = {}
        with open(queries_path, encoding='utf-8') as f:
            for qid, qtext in csv.reader(f, delimiter='\t'):
                self.queries[qid] = qtext.strip()

        self.passages = {}
        with open(corpus_path, encoding='utf-8') as f:
            for line in f:
                pid, ptext = line.strip().split('\t', 1)
                self.passages[pid] = ptext.strip()

    def _encode(self, text):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __iter__(self):
        with open(self.triples_path, encoding='utf-8') as f:
            for qid, pos_pid, neg_pid in csv.reader(f, delimiter='\t'):
                if qid in self.queries and pos_pid in self.passages and neg_pid in self.passages:
                    qtext = self.queries[qid]
                    pos_text = self.passages[pos_pid]
                    neg_text = self.passages[neg_pid]

                    yield {
                        "query": self._encode(qtext),
                        "pos_doc": self._encode(pos_text),
                        "neg_doc": self._encode(neg_text)
                    }
