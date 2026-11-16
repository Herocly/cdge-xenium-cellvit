"""Check that class-11 pixels cannot influence the CellViT type loss."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integrated_pipeline.train import _imports, _losses, _type_supervision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdge-root", type=Path, default=ROOT / "vendor/CDGE")
    args = parser.parse_args()
    torch, _, _, classes = _imports(args.cdge_root)
    torch.manual_seed(19)
    probs = torch.softmax(torch.randn(1, 6, 4, 4), dim=1).requires_grad_()
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    targets[0, 0, 0] = 11
    targets[0, 1, 1] = 1
    losses = _losses(classes)
    first = _type_supervision(torch, probs, targets, losses)
    first.backward()
    if probs.grad is None or not torch.all(probs.grad[:, :, 0, 0] == 0):
        raise AssertionError("Unknown pixel contributed to type-loss gradient")
    changed = probs.detach().clone()
    changed[:, :, 0, 0] = torch.softmax(torch.randn(6), dim=0)
    second = _type_supervision(torch, changed, targets, losses)
    if not torch.allclose(first.detach(), second.detach(), atol=1e-7):
        raise AssertionError("Unknown pixel changed the type loss")
    print("PASSED: Unknown pixel has zero type-loss gradient and no effect on type loss")


if __name__ == "__main__":
    main()
