import wandb
import argparse

import torch
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model import EnergyBasedDenseIR
from dataset import MSMARCODataModule


def train(args):

    # -- hyperparams
    BATCH_SIZE = args.batch_size
    MARGIN = args.margin
    LEARNING_RATE = args.lr
    MAX_STEPS = args.max_steps
    MAX_EPOCHS = args.max_epochs
    SHARED_ENCODER = args.shared_encoder
    TRAINABLE_ENCODER = args.trainable_encoder
    ENCODER = args.encoder
    DEVICES = args.devices
    NUM_WORKERS = args.num_workers
    WANDB_NAME = args.wandb_name
    STRATEGY = args.strategy

    # -- standardize the randomness
    seed_everything(786, workers=True)

    # -- wandb
    wandb_logger = WandbLogger(
        project="playground",
        name=WANDB_NAME,
        log_model="all",
        save_dir="./wandb_logs",
    )

    wandb_logger.experiment.config.update({
        "batch_size": BATCH_SIZE,
        "shared_encoder": SHARED_ENCODER,
        "margin": MARGIN,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "trainable_encoder": TRAINABLE_ENCODER,
        "encoder": ENCODER,
    })

    # -- callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="./checkpoints"+f"/{WANDB_NAME}",
        filename="ebt-step{step:06d}",
        every_n_train_steps=2000,     
        save_top_k=-1,              
        save_last=True,             
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # -- data module
    dm = MSMARCODataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # -- model
    model = EnergyBasedDenseIR(margin=MARGIN, lr=LEARNING_RATE, encoder_name=ENCODER, trainable_encoder=TRAINABLE_ENCODER, shared_encoder=SHARED_ENCODER)
    wandb_logger.watch(model, log="gradients", log_freq=1000)

    # -- trainer
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=DEVICES,
        strategy=STRATEGY,
        precision="16-mixed",
        max_steps=MAX_STEPS,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        limit_val_batches=0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=0,
    )

    # -- here we goooo
    trainer.fit(model, datamodule=dm)

    wandb.finish()


# -- argparser
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--shared_encoder", action="store_true", default=False)
    parser.add_argument("--trainable_encoder", action="store_true", default=False)
    parser.add_argument("--encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--strategy", type=str, default="ddp")

    args = parser.parse_args()

    # derived name
    args.wandb_name = (
        f"dense_ir_{args.encoder.replace('/', '_')}_"
        f"ebm_margin_{args.margin}_lr_{args.lr}_"
        f"bs_{args.batch_size}_"
        f"shared_encoder_{args.shared_encoder}_"
        f"trainable_encoder_{args.trainable_encoder}"
    )

    return args


if __name__ == "__main__":
    train(parse_args())