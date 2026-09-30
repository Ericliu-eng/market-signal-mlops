import argparse
from collections.abc import Sequence

from market_signal_mlops.registry.service import RegistryService


DEFAULT_MODEL_NAME = "market-signal-classifier"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Market Signal model aliases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Move the champion alias to an earlier model version.",
    )
    rollback_parser.add_argument(
        "--version",
        required=True,
        help="Model version that should become champion.",
    )
    rollback_parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Registered model name.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "rollback":
        service = RegistryService(args.model_name)
        service.rollback_champion(version=args.version)
        print(
            f"Champion alias for {args.model_name} "
            f"now points to version {args.version}."
        )


if __name__ == "__main__":
    main()
