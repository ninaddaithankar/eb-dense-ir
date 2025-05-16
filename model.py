import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

from torchmetrics.classification import BinaryAccuracy


class HFEncoder(nn.Module):
    """Wrap a HuggingFace transformer and return the CLS embedding."""

    def __init__(self, model_name: str = "bert-base-uncased", trainable: bool = True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if not trainable:
            self.transformer.requires_grad_(False)

    @property
    def hidden_size(self):
        return self.transformer.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,  **kwargs):
        # outputs.last_hidden_state -> (B, L, H)
        # CLS token is at position 0 by convention for BERT‑style models
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # (B, H)


class EnergyScorer(nn.Module):
    """Simple 2-layer MLP that maps [q_emb || d_emb] -> scalar energy."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([q_emb, d_emb], dim=-1)  # (B, 2H)
        return self.scorer(joint).squeeze(-1)      # (B,)


class EnergyBasedDenseIR(pl.LightningModule):
    """PyTorch-Lightning module for training an Energy-Based Transformer retriever."""

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        lr: float = 2e-5,
        margin: float = 1.0,
        shared_encoder: bool = True,
        trainable_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoders -----------------------------------------------------------
        if shared_encoder:
            shared = HFEncoder(encoder_name, trainable=trainable_encoder)
            self.query_encoder = shared
            self.doc_encoder = shared
        else:
            self.query_encoder = HFEncoder(encoder_name, trainable=trainable_encoder)
            self.doc_encoder = HFEncoder(encoder_name, trainable=trainable_encoder)

        hidden_dim = self.query_encoder.hidden_size
        self.energy_scorer = EnergyScorer(hidden_dim)

        # Hyper‑parameters ---------------------------------------------------
        self.margin = margin
        self.lr = lr

        # Metric: pairwise accuracy (pos energy lower than neg)
        # self.val_pair_acc = BinaryAccuracy()

    def forward(self, q_batch: dict, d_batch: dict) -> torch.Tensor:
        q_emb = self.query_encoder(**q_batch)
        d_emb = self.doc_encoder(**d_batch)
        return self.energy_scorer(q_emb, d_emb)  # (B,)

    def _hinge_loss(self, pos_energy: torch.Tensor, neg_energy: torch.Tensor) -> torch.Tensor:
        return F.relu(pos_energy - neg_energy + self.margin).mean()

    def training_step(self, batch, batch_idx):
        pos_energy = self(batch["query"], batch["pos_doc"])
        neg_energy = self(batch["query"], batch["neg_doc"])
        loss = self._hinge_loss(pos_energy, neg_energy)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/positive_energy", pos_energy.mean(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/negative_energy", neg_energy.mean(), prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pos_energy = self(batch["query"], batch["pos_doc"])
        neg_energy = self(batch["query"], batch["neg_doc"])
        loss = self._hinge_loss(pos_energy, neg_energy)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # pairwise accuracy metric
        acc = (pos_energy < neg_energy).float().mean()
        self.log("val/pair_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

