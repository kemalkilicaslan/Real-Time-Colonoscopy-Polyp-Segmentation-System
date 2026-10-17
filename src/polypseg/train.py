"""Hand-written PyTorch training loop (no trainer framework).

    python -m polypseg.train --config configs/unet_r34.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SegmentationDataset, group_split, kvasir_samples, negative_samples
from .model import build_model, load_config, pick_device, save_checkpoint
from .preprocess import eval_transforms, train_transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by dependencies
    torch.manual_seed(seed)


class DiceBCELoss(nn.Module):
    """BCE handles per-pixel calibration, soft Dice counters the foreground/background imbalance.

    The Dice term is averaged over images that contain a polyp only. On an empty mask soft Dice
    is ``eps / (sum(p) + eps)``: a handful of uncertain pixels already costs ~1, which swamps the
    loss and destabilised training once polyp-free frames were added. BCE alone teaches negatives.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight, self.dice_weight, self.eps = bce_weight, dice_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce_weight * self.bce(logits, target)
        pos = target.sum(dim=(1, 2, 3)) > 0
        if pos.any():
            prob, tgt = torch.sigmoid(logits[pos]), target[pos]
            inter = (prob * tgt).sum(dim=(1, 2, 3))
            denom = prob.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
            soft_dice = (2 * inter + self.eps) / (denom + self.eps)
            loss = loss + self.dice_weight * (1 - soft_dice).mean()
        return loss


@torch.no_grad()
def batch_dice(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Hard per-image Dice at network resolution."""
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))


@torch.no_grad()
def epoch_scores(dice: torch.Tensor, pred_frac: torch.Tensor, is_pos: torch.Tensor) -> dict:
    """Model-selection score that does not let one class dominate.

    pos_dice:  mean Dice over images with a polyp
    neg_clean: share of polyp-free images where < 0.1 % of the frame is predicted as polyp
    score:     their mean (just pos_dice when there are no negatives)
    """
    pos_dice = dice[is_pos].mean().item() if is_pos.any() else float("nan")
    if (~is_pos).any():
        neg_clean = (pred_frac[~is_pos] < 1e-3).float().mean().item()
        score = (pos_dice + neg_clean) / 2
    else:
        neg_clean, score = float("nan"), pos_dice
    return {"pos_dice": pos_dice, "neg_clean": neg_clean, "score": score}


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss, dices, fracs, pos = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            logits = logits.detach()
            dices.append(batch_dice(logits, y).cpu())
            fracs.append((torch.sigmoid(logits) > 0.5).float().mean(dim=(1, 2, 3)).cpu())
            pos.append((y.sum(dim=(1, 2, 3)) > 0).cpu())
    scores = epoch_scores(torch.cat(dices), torch.cat(fracs), torch.cat(pos))
    return {"loss": total_loss / len(loader.dataset), **scores}


def train(cfg: dict, out_root: str | Path = "runs", device: str | None = None) -> Path:
    seed_everything(cfg["seed"])
    dev = pick_device(device)
    out = Path(out_root) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    samples = kvasir_samples(d["kvasir_root"])
    train_s, val_s = group_split(samples, d["val_fraction"], cfg["seed"])
    split = {"train": [s.name for s in train_s], "val": [s.name for s in val_s]}
    if d.get("negatives_root"):
        neg_train, neg_val = group_split(
            negative_samples(d["negatives_root"]), d["val_fraction"], cfg["seed"]
        )
        split |= {"neg_train": [s.name for s in neg_train], "neg_val": [s.name for s in neg_val]}
        print(f"negatives: train={len(neg_train)}  val={len(neg_val)}")
        train_s, val_s = train_s + neg_train, val_s + neg_val
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(f"device={dev}  train={len(train_s)}  val={len(val_s)}")

    loader_kw = dict(
        batch_size=t["batch_size"],
        num_workers=t["num_workers"],
        persistent_workers=t["num_workers"] > 0,
        pin_memory=dev.type == "cuda",
    )
    train_dl = DataLoader(
        SegmentationDataset(train_s, train_transforms(d["image_size"])),
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_dl = DataLoader(SegmentationDataset(val_s, eval_transforms(d["image_size"])), **loader_kw)

    model = build_model(m["arch"], m["encoder"], m["encoder_weights"]).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"], eta_min=1e-6)
    loss_fn = DiceBCELoss(t["bce_weight"], t["dice_weight"])

    best, best_epoch = -1.0, -1
    cols = ["loss", "pos_dice", "neg_clean", "score"]
    with open(out / "history.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "lr", *[f"train_{c}" for c in cols], *[f"val_{c}" for c in cols],
                      "seconds"])  # fmt: skip
        for epoch in range(1, t["epochs"] + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr = run_epoch(model, train_dl, loss_fn, dev, optimizer)
            va = run_epoch(model, val_dl, loss_fn, dev)
            scheduler.step()
            secs = time.time() - t0
            log.writerow([epoch, f"{lr:.2e}", *[f"{tr[c]:.4f}" for c in cols],
                          *[f"{va[c]:.4f}" for c in cols], f"{secs:.1f}"])  # fmt: skip
            f.flush()
            flag = ""
            if va["score"] > best:
                best, best_epoch, flag = va["score"], epoch, " *"
                save_checkpoint(out / "best.pt", model, cfg, epoch=epoch, val=va)
            neg = f"  neg_clean {va['neg_clean']:.3f}" if va["neg_clean"] == va["neg_clean"] else ""
            print(f"ep {epoch:3d}  loss {tr['loss']:.3f}/{va['loss']:.3f}  "
                  f"dice {tr['pos_dice']:.3f}/{va['pos_dice']:.3f}{neg}  {secs:.0f}s{flag}",
                  flush=True)  # fmt: skip
            if epoch - best_epoch >= t["early_stopping_patience"]:
                print(f"early stop: no improvement for {t['early_stopping_patience']} epochs")
                break
    print(f"best val score {best:.4f} at epoch {best_epoch} -> {out / 'best.pt'}")
    return out / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, args.out, args.device)


if __name__ == "__main__":
    main()
