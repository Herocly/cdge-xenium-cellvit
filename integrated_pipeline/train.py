"""CellViT-256 training with a masked type loss for Unknown nuclei.

The model and the binary/HV/type losses are from the supplied CellViT code.
The ONLY loss adaptation is that class-11 pixels are excluded from type loss;
the original six-class trainer cannot one-hot encode class 11. Segmentation
losses still see their genuine nucleus geometry. This adaptation must be used
for both baseline and proposed label sources in a controlled comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


def _imports(cdge_root: Path):
    sys.path.insert(0, str(cdge_root / "CellViT"))
    import torch
    from cell_segmentation.datasets.pannuke import PanNukeDataset
    from models.segmentation.cell_segmentation.cellvit_shared import CellViT256Shared
    from base_ml.base_loss import (FocalTverskyLoss, DiceLoss, MSELossMaps,
                                   MSGELossMaps, XentropyLoss, MCFocalTverskyLoss)
    return torch, PanNukeDataset, CellViT256Shared, (FocalTverskyLoss, DiceLoss, MSELossMaps,
                                                      MSGELossMaps, XentropyLoss, MCFocalTverskyLoss)


def _model(CellViT256Shared, config: dict, device):
    settings = config.get("cellvit", {})
    model = CellViT256Shared(
        model256_path=None, num_nuclei_classes=6, num_tissue_classes=19,
        drop_rate=0, attn_drop_rate=0.1, drop_path_rate=0.1,
    )
    checkpoint = settings.get("pretrained_checkpoint")
    if checkpoint:
        import torch
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = obj.get("model_state_dict", obj.get("model", obj.get("state_dict", obj)))
        model.load_state_dict(state, strict=True)
    # Teacher's ViT256 experiment freezes the encoder at initialization.
    model.freeze_encoder()
    return model.to(device)


def _losses(classes):
    Focal, Dice, MSE, MSGE, CE, MCFocal = classes
    return Focal(), Dice(), MSE(), MSGE(), CE(), MCFocal(num_classes=6)


def _type_supervision(torch, pred_prob, type_ids, loss_objects):
    """Apply original type losses only to pixels with known labels (0..5)."""
    import torch.nn.functional as F
    valid = type_ids != 11
    if not torch.any(valid):
        return pred_prob.sum() * 0
    if torch.any((type_ids[valid] < 0) | (type_ids[valid] >= 6)):
        raise ValueError("Unexpected type ID outside CellViT's six classes")
    # Gather known locations into a dense tensor. The original loss functions
    # then see exactly the same probabilities/targets as on an all-known map.
    flat_prob = pred_prob.permute(0, 2, 3, 1)[valid].T[None, :, None, :]
    flat_type = F.one_hot(type_ids[valid].long(), num_classes=6).float().T[None, :, None, :]
    _, dice, _, _, ce, mc_focal = loss_objects
    return 0.5 * ce(flat_prob, flat_type) + 0.2 * dice(flat_prob, flat_type) + 0.5 * mc_focal(flat_prob, flat_type)


def _loss(torch, raw, masks, tissue_names, tissue_to_id: dict, objects, device):
    import torch.nn.functional as F
    focal, dice, mse, msge, _, _ = objects
    binary = masks["nuclei_binary_map"].to(device)
    binary_gt = F.one_hot(binary, num_classes=2).permute(0, 3, 1, 2).float()
    binary_prob = F.softmax(raw["nuclei_binary_map"], dim=1)
    hv_gt = masks["hv_map"].to(device)
    type_prob = F.softmax(raw["nuclei_type_map"], dim=1)
    type_ids = masks["nuclei_type_map"].to(device)
    tissue_aliases = {"Ovary": "Ovarian", "Pancreas": "Pancreatic", "Bowel": "Colon"}
    tissue_ids = torch.tensor([tissue_to_id[tissue_aliases.get(str(x), str(x))] for x in tissue_names], device=device)
    binary_loss = focal(binary_prob, binary_gt) + dice(binary_prob, binary_gt)
    hv_loss = 2.5 * mse(raw["hv_map"], hv_gt) + 8 * msge(raw["hv_map"], hv_gt, binary_gt, device)
    type_loss = _type_supervision(torch, type_prob, type_ids, objects)
    tissue_loss = 0.1 * F.cross_entropy(raw["tissue_types"], tissue_ids)
    total = binary_loss + hv_loss + type_loss + tissue_loss
    return total, {"binary": float(binary_loss.detach()), "hv": float(hv_loss.detach()),
                   "type": float(type_loss.detach()), "tissue": float(tissue_loss.detach())}


def _dataset_config(dataset: Path):
    import yaml
    conf = yaml.safe_load((dataset / "dataset_config.yaml").read_text())
    if conf["nuclei_types"] != {"Background": 0, "Neoplastic": 1, "Inflammatory": 2,
                                 "Connective": 3, "Dead": 4, "Epithelial": 5}:
        raise ValueError("CellViT class mapping does not match the expected mapping")
    return conf


def train(config: dict, dataset: Path, run_dir: Path, smoke: bool = False) -> dict:
    torch, PanNukeDataset, Model, classes = _imports(config["cdge_root"])
    from torch.utils.data import DataLoader
    settings = config.get("cellvit", {})
    seed = int(settings.get("random_seed", 19))
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = settings.get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    meta = _dataset_config(dataset)
    train_set = PanNukeDataset(dataset, folds=[0])
    val_set = PanNukeDataset(dataset, folds=[1])
    if not train_set or not val_set:
        raise ValueError("Train and Validation folds must both be nonempty")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=int(settings.get("batch_size", 1)), shuffle=True, num_workers=0,
                              generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    model = _model(Model, config, device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=float(settings.get("learning_rate", 3e-4)),
                                  betas=(0.85, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(settings.get("scheduler_gamma", 0.85)))
    losses = _losses(classes)
    out = run_dir / "cellvit_training"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    epochs = 1 if smoke else int(settings.get("epochs", 130))
    if epochs < 1:
        raise ValueError("cellvit.epochs must be positive")
    best_val, best_epoch = math.inf, -1
    for epoch in range(epochs):
        if not smoke and epoch == int(settings.get("unfreeze_epoch", 25)):
            model.unfreeze_encoder()
            present = {id(p) for group in optimizer.param_groups for p in group["params"]}
            newly_trainable = [p for p in model.parameters() if p.requires_grad and id(p) not in present]
            if newly_trainable:
                optimizer.add_param_group({"params": newly_trainable, "lr": optimizer.param_groups[0]["lr"]})
        model.train()
        for step, (images, masks, tissues, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(images)
            loss, parts = _loss(torch, raw, masks, tissues, meta["tissue_types"], losses, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not params or not all(torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("Missing or nonfinite gradients")
            before = params[0].detach().clone() if smoke else None
            optimizer.step()
            if smoke and torch.equal(before, params[0].detach()):
                raise RuntimeError("Optimizer did not update a trainable parameter")
            records.append({"epoch": epoch, "step": step, "loss": float(loss.detach()), **parts})
            if smoke:
                break
        if smoke:
            break
        model.eval()
        with torch.no_grad():
            val_losses = []
            for images, masks, tissues, _ in val_loader:
                raw = model(images.to(device))
                val_loss, _ = _loss(torch, raw, masks, tissues, meta["tissue_types"], losses, device)
                val_losses.append(float(val_loss))
        mean_val = sum(val_losses) / len(val_losses)
        records.append({"epoch": epoch, "validation_loss": mean_val})
        if mean_val < best_val:
            best_val, best_epoch = mean_val, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "class_mapping": meta["nuclei_types"], "unknown_policy": "mask type loss only",
                        "seed": seed}, out / "model_best.pth")
        if epoch - best_epoch >= int(settings.get("early_stopping_patience", 20)):
            records.append({"epoch": epoch, "stopped": "early_stopping"})
            break
        scheduler.step()
    checkpoint = out / ("smoke_checkpoint.pth" if smoke else "model_last.pth")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "class_mapping": meta["nuclei_types"], "unknown_policy": "mask type loss only"}, checkpoint)
    reloaded = _model(Model, {"cellvit": {}}, device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        sample_image, _, _, _ = val_set[0]
        prediction = reloaded(sample_image[None].to(device))
    result = {"status": "passed", "checkpoint": str(checkpoint), "best_validation_epoch": best_epoch,
              "random_seed": seed, "steps": len([r for r in records if "step" in r]),
              "reload_prediction_shape": list(prediction["nuclei_type_map"].shape), "records": records}
    (out / ("smoke_report.json" if smoke else "training_report.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate(config: dict, dataset: Path, run_dir: Path) -> dict:
    """Conditional class evaluation on GT nuclei; not detector/instance AP."""
    torch, PanNukeDataset, Model, _ = _imports(config["cdge_root"])
    external = config.get("cellvit", {}).get("external_test_dataset")
    if not external:
        raise ValueError("Set cellvit.external_test_dataset for independent evaluation")
    external = Path(external)
    if not external.is_dir():
        raise FileNotFoundError(external)
    meta = _dataset_config(dataset)
    test_set = PanNukeDataset(external, folds=config.get("cellvit", {}).get("test_folds", [0, 1, 2]))
    device = torch.device(config.get("cellvit", {}).get("device", "cuda"))
    model = _model(Model, {"cellvit": {}}, device)
    checkpoint = run_dir / "cellvit_training/model_best.pth"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    counts = {i: {"tp": 0, "fp": 0, "fn": 0} for i in (1, 2, 3, 5)}
    ignored_dead = 0
    with torch.no_grad():
        for image, masks, _, _ in test_set:
            logits = model(image[None].to(device))["nuclei_type_map"]
            prob = torch.softmax(logits, dim=1)[0].cpu()
            inst = masks["instance_map"]
            true = masks["nuclei_type_map"]
            for nucleus_id in torch.unique(inst):
                if int(nucleus_id) == 0:
                    continue
                region = inst == nucleus_id
                target = int(torch.mode(true[region]).values)
                if target == 4:
                    ignored_dead += 1; continue
                if target not in counts:
                    continue
                predicted = int(prob[:, region].mean(dim=1).argmax())
                for cls in counts:
                    counts[cls]["tp"] += int(target == cls and predicted == cls)
                    counts[cls]["fp"] += int(target != cls and predicted == cls)
                    counts[cls]["fn"] += int(target == cls and predicted != cls)
    f1 = {str(cls): (2*c["tp"] / (2*c["tp"] + c["fp"] + c["fn"]))
          if 2*c["tp"] + c["fp"] + c["fn"] else None for cls, c in counts.items()}
    result = {"metric_scope": "type classification conditional on reference nucleus instances",
              "reference": str(external), "dead_nuclei_excluded": ignored_dead,
              "per_class_f1": f1, "macro_f1": sum(v for v in f1.values() if v is not None) / sum(v is not None for v in f1.values()),
              "counts": counts}
    out = run_dir / "cellvit_evaluation"; out.mkdir(parents=True, exist_ok=True)
    (out / "conditional_type_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate", "smoke"), required=True)
    args = parser.parse_args()
    config = json.loads(args.resolved_config.read_text())
    config["cdge_root"] = Path(config["cdge_root"])
    dataset = args.run_dir / "cellvit_dataset"
    outcome = (evaluate(config, dataset, args.run_dir) if args.stage == "evaluate" else
               train(config, dataset, args.run_dir, smoke=args.stage == "smoke"))
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
