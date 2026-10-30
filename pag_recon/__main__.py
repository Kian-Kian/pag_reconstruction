"""Command-line entry points for grain reconstruction and training."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prior-austenite grain reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser(
        "reconstruct", help="Reconstruct one alpha CTF using the frozen ensemble")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--models", type=Path, default=Path("models"))
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    infer.add_argument("--alpha-phase-id", type=int, default=1)
    infer.add_argument("--parent-lattice-angstrom", type=float, default=3.66)
    example = commands.add_parser(
        "example", help="Generate synthetic alpha and paired training CTFs")
    example.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare-training", help="Build paired graphs, fit OR and prepare fold features")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.command == "reconstruct":
        from .pipeline import reconstruct
        print(json.dumps(reconstruct(args.input, args.models, args.output, device=args.device,
                                     alpha_phase_id=args.alpha_phase_id, parent_lattice_angstrom=args.parent_lattice_angstrom), indent=2))
    elif args.command == "example":
        from .synthetic import generate_example
        generate_example(args.output)
        print(args.output)
    else:
        from .preparation import prepare_training
        prepare_training(args.data, args.output, folds=args.folds)


if __name__ == "__main__":
    main()
