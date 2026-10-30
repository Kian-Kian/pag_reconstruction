"""Train MLP and GNN2 models for each fold and save their inference checkpoints."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-epochs", type=int,
                        help="Override both epoch limits for a short execution check")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new training output directory")
    records = json.loads((args.prepared / "folds.json").read_text())["records"]
    args.output.mkdir(parents=True)
    bundle = args.output / "models"
    bundle.mkdir()
    for record in records:
        fold = int(record["fold"])
        name = f"fold_{fold:02d}"
        prepared = args.prepared / name
        run = args.output / name
        folder = bundle / name
        folder.mkdir()
        shutil.copyfile(prepared / "orientation_relationship.json",
                        folder / "orientation_relationship.json")
        for model in ("mlp", "gnn2"):
            command = [sys.executable, "-B", "-m", "pag_recon.training",
                       "--model", model, "--graph-root", str(prepared / "variant_graphs"),
                       "--split-manifest", str(prepared / "split.json"),
                       "--checkpoint-root", str(run / "checkpoints"),
                       "--log-root", str(run / "logs"), "--seed", str(42+fold),
                       "--device", args.device,
                       "--num-workers", str(args.num_workers)]
            if args.smoke_epochs is not None:
                if args.smoke_epochs < 1:
                    raise ValueError("--smoke-epochs must be positive")
                command += ["--epochs", str(args.smoke_epochs)]
            subprocess.run(command, check=True)
            shutil.copyfile(run / "checkpoints" / model / "inference.pt", folder / f"{model}.pt")
            shutil.copyfile(run / "logs" / model / "feature_normalizer.json",
                            folder / f"{model}_normalizer.json")
    shutil.copyfile(args.prepared / "common_orientation_relationship.json",
                    bundle / "common_orientation_relationship.json")
    (args.output / "training_run.json").write_text(json.dumps({"folds": len(records),
                                                               "smoke_epochs": args.smoke_epochs, "full_training_schedule": args.smoke_epochs is None,
                                                               "inference_bundle": "models", "five_fold_inference_compatible": len(records) == 5}, indent=2)+"\n")


if __name__ == "__main__":
    main()
