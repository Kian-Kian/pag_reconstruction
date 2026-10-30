"""Train physics-residual MLP or edge-aware GNN models."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from pag_recon.support.experiment import experiment_manifest, write_experiment_manifest
from pag_recon.graph.io import FeatureNormalizer, graph_path, load_split_manifest
from pag_recon.graph.metrics import (
    average_precision,
    balanced_boundary_loss,
    best_f1_threshold,
    roc_auc,
)
from pag_recon.graph.physics import (
    PhysicsGraphDataset,
    PhysicsResidualEdgeClassifier,
    collate_physics_graphs,
    fit_physics_normalizer,
    triangle_transitivity_loss,
)
from pag_recon.support.runtime import json_dump, save_checkpoint, seed_everything
from pag_recon.support.defaults import (
    CHECKPOINT_ROOT,
    LOG_ROOT,
    PHYSICS_BOUNDARY_DEGREES,
    PHYSICS_PROBABILITY_SCALE_DEGREES,
    RANDOM_SEED,
    SPLIT_MANIFEST,
    VARIANT_GRAPH_ROOT,
)


@dataclass
class Config:
    device: str
    graph_root: str
    split_manifest: str
    checkpoint_root: str
    log_root: str
    run_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_channels: int
    layers: int
    dropout: float
    num_workers: int
    seed: int
    physics_angle_threshold: float
    physics_probability_scale: float
    residual_weight: float
    triangle_weight: float
    calibration_weight: float
    prediction_mode: str
    resume: str | None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a physics-logit residual edge MLP/GNN")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--graph-root",
        default=str(VARIANT_GRAPH_ROOT),
    )
    parser.add_argument(
        "--split-manifest",
        default=str(SPLIT_MANIFEST),
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(CHECKPOINT_ROOT),
    )
    parser.add_argument(
        "--log-root", default=str(LOG_ROOT)
    )
    parser.add_argument("--model", choices=("mlp", "gnn2"), required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--physics-angle-threshold", type=float, default=PHYSICS_BOUNDARY_DEGREES)
    parser.add_argument("--physics-probability-scale", type=float,
                        default=PHYSICS_PROBABILITY_SCALE_DEGREES)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--triangle-weight", type=float, default=0.05)
    parser.add_argument("--calibration-weight", type=float, default=0.05)
    parser.add_argument("--resume")
    options = vars(parser.parse_args())
    model = options.pop("model")
    defaults = {"mlp": (96, 0, 40, 3e-4), "gnn2": (128, 2, 120, 1e-3)}
    hidden, layers, epochs, rate = defaults[model]
    for key, value in {"hidden_channels": hidden, "epochs": epochs,
                       "learning_rate": rate, "run_name": model}.items():
        if options[key] is None:
            options[key] = value
    options.update(layers=layers, prediction_mode="residual")
    return Config(**options)


def class_counts(paths: list[Path]) -> tuple[int, int]:
    boundary = same = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            label = data["edge_same_parent"]
        valid = label >= 0
        boundary += int(np.count_nonzero(label[valid] == 0))
        same += int(np.count_nonzero(label[valid] == 1))
    return boundary, same


def loss_terms(batch, logits, residual, pos_weight: float, config: Config) -> dict[str, torch.Tensor]:
    valid = batch.valid_edge_mask
    supervised = balanced_boundary_loss(logits, batch.boundary_target, valid, pos_weight)
    residual_loss = torch.square(residual[valid]).mean()
    calibration = functional.mse_loss(torch.sigmoid(logits[valid]), batch.boundary_target[valid])
    triangle_valid = batch.valid_edge_mask[batch.triangle_edge_index].all(dim=0)
    triangle = triangle_transitivity_loss(logits, batch.triangle_edge_index[:, triangle_valid])
    total = (
        supervised
        + config.residual_weight * residual_loss
        + config.triangle_weight * triangle
        + config.calibration_weight * calibration
    )
    return {
        "total": total,
        "supervised": supervised,
        "residual": residual_loss,
        "triangle": triangle,
        "calibration": calibration,
    }


def run_epoch(model, loader, device, pos_weight, config, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "supervised", "residual", "triangle", "calibration")}
    edge_count = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits, residual = model(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
                batch.physics_boundary_logit,
            )
            losses = loss_terms(batch, logits, residual, pos_weight, config)
        if training:
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        count = int(batch.valid_edge_mask.sum().item())
        edge_count += count
        for name, value in losses.items():
            totals[name] += float(value.item()) * count
        if not training:
            valid = batch.valid_edge_mask
            probabilities.append(torch.sigmoid(logits[valid]).float().cpu().numpy())
            targets.append(batch.boundary_target[valid].cpu().numpy())
    result = {name: value / max(edge_count, 1) for name, value in totals.items()}
    if not training:
        probability = np.concatenate(probabilities)
        target = np.concatenate(targets)
        threshold, metrics = best_f1_threshold(probability, target)
        result.update(
            {
                "average_precision": average_precision(probability, target),
                "roc_auc": roc_auc(probability, target),
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "threshold": threshold,
            }
        )
    return result


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)
    manifest = load_split_manifest(config.split_manifest)
    if not manifest["train"] or not manifest["val"] or set(manifest["train"]) & set(manifest["val"]):
        raise ValueError("Training and validation maps must be nonempty and disjoint")
    train_paths = [graph_path(config.graph_root, key) for key in manifest["train"]]
    resume = torch.load(config.resume, map_location=device,
                        weights_only=False) if config.resume else None
    normalizer = (
        FeatureNormalizer.from_dict(resume["feature_normalizer"])
        if resume is not None
        else fit_physics_normalizer(train_paths)
    )
    dataset_kwargs = {
        "physics_angle_threshold_degrees": config.physics_angle_threshold,
        "physics_probability_scale_degrees": config.physics_probability_scale,
    }
    train_dataset = PhysicsGraphDataset(
        config.graph_root, manifest["train"], normalizer, **dataset_kwargs)
    val_dataset = PhysicsGraphDataset(
        config.graph_root, manifest["val"], normalizer, **dataset_kwargs)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
        "collate_fn": collate_physics_graphs,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    boundary_count, same_count = class_counts(train_paths)
    if boundary_count == 0 or same_count == 0:
        raise ValueError("Training requires labeled boundary and same-parent edges")
    pos_weight = same_count / max(boundary_count, 1)
    if config.layers < 0:
        raise ValueError("--layers must be non-negative")
    model = PhysicsResidualEdgeClassifier(
        config.hidden_channels, config.layers, config.dropout, config.prediction_mode
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_ap = -1.0
    if resume is not None:
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        start_epoch = int(resume["epoch"]) + 1
        best_ap = float(resume["best_average_precision"])
        random.setstate(resume["python_random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_random_state"].cpu())

    checkpoint_dir = Path(config.checkpoint_root) / config.run_name
    log_dir = Path(config.log_root) / config.run_name
    if config.resume is None and (checkpoint_dir.exists() or log_dir.exists()):
        raise FileExistsError("Use a new output directory for each training run")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    json_dump(asdict(config), log_dir / "config.json")
    json_dump(normalizer.as_dict(), log_dir / "feature_normalizer.json")
    history_path = log_dir / "history.csv"
    fields = [
        "epoch", "train_total", "val_total", "val_supervised", "val_residual", "val_triangle",
        "val_calibration", "val_average_precision", "val_roc_auc", "val_f1", "val_threshold",
        "learning_rate", "seconds",
    ]
    with history_path.open("a" if start_epoch > 1 else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if start_epoch == 1:
            writer.writeheader()
        for epoch in range(start_epoch, config.epochs + 1):
            started = time.perf_counter()
            train = run_epoch(model, train_loader, device, pos_weight, config, optimizer, scaler)
            validation = run_epoch(model, val_loader, device, pos_weight, config)
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_total": train["total"],
                **{f"val_{name}": validation[name] for name in ("total", "supervised", "residual", "triangle", "calibration", "average_precision", "roc_auc", "f1", "threshold")},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch:03d} train={train['total']:.4f} val={validation['total']:.4f} "
                f"AP={validation['average_precision']:.4f} AUC={validation['roc_auc']:.4f} "
                f"F1={validation['f1']:.4f}@{validation['threshold']:.2f} "
                f"tri={validation['triangle']:.4f} residual={validation['residual']:.4f} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )
            state = {
                "epoch": epoch,
                "model_type": (
                    "physics_residual_mlp" if config.layers == 0 else "physics_residual_gnn"
                ),
                "model_kwargs": {
                    "hidden_channels": config.hidden_channels,
                    "layers": config.layers,
                    "dropout": config.dropout,
                    "prediction_mode": config.prediction_mode,
                },
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "feature_normalizer": normalizer.as_dict(),
                "edge_boundary_threshold": validation["threshold"],
                "best_average_precision": max(best_ap, validation["average_precision"]),
                "config": asdict(config),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
            }
            save_checkpoint(checkpoint_dir / "last.pt", **state)
            if validation["average_precision"] > best_ap:
                best_ap = validation["average_precision"]
                state["best_average_precision"] = best_ap
                save_checkpoint(checkpoint_dir / "best.pt", **state)
                inference = {key: state[key] for key in (
                    "epoch", "model_kwargs", "model_state", "feature_normalizer", "config"
                )}
                save_checkpoint(checkpoint_dir / "inference.pt", **inference)

    record = experiment_manifest(
        experiment_name=config.run_name,
        command=sys.argv,
        config=asdict(config),
        source_files=(Path(__file__), Path(__file__).parent / "graph" / "physics.py"),
        data_files=(config.split_manifest, checkpoint_dir / "best.pt"),
        results={"status": "completed", "best_validation_average_precision": best_ap},
    )
    write_experiment_manifest(record, log_dir / "experiment_manifest.json")


if __name__ == "__main__":
    main()
