"""
Single entry point for all corpus operations.

Usage::

    python -m dataset_downloader download voxceleb2 /data/vox2 --num-jobs 16 --verify
    python -m dataset_downloader prepare voxceleb1 /data/raw /data/prepared --vad
    python -m dataset_downloader list
"""

import argparse
import logging
import sys
from pathlib import Path

from . import REGISTRY
from .pipeline import DownloadPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


# ── Programmatic API ──────────────────────────────────────────────


def download_dataset(
    name: str,
    target_dir: str,
    force: bool = False,
    verify: bool = False,
    num_jobs: int = 4,
) -> None:
    """
    Programmatic download entry point.

    Example::

        from dataset_downloader.cli import download_dataset
        download_dataset("voxceleb2", "/data/vox2", verify=True, num_jobs=16)
    """

    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset: '{name}'. Available: {list(REGISTRY)}")

    spec = REGISTRY[name]
    DownloadPipeline().run(
        spec, Path(target_dir), force=force, verify=verify, num_jobs=num_jobs
    )


# ── CLI ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_downloader",
        description="Télécharge et prépare un corpus audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command")

    # --- Sous-commande : download ---
    dl = sub.add_parser("download", help="Télécharge un corpus.")
    dl.add_argument(
        "--list", action="store_true", help="Affiche les corpus disponibles."
    )
    dl.add_argument("dataset", nargs="?", choices=list(REGISTRY), metavar="DATASET")
    dl.add_argument("target_dir", nargs="?", metavar="TARGET_DIR")
    dl.add_argument("--force", action="store_true", default=False)
    dl.add_argument("--verify", action="store_true", default=False)
    dl.add_argument("--num-jobs", type=int, default=4, dest="num_jobs")

    # --- Sous-commande : prepare ---
    pr = sub.add_parser("prepare", help="Prépare les données d'un corpus téléchargé.")
    pr.add_argument("dataset", choices=list(REGISTRY), metavar="DATASET")
    pr.add_argument("in_data", metavar="IN_DATA", help="Répertoire des données brutes.")
    pr.add_argument(
        "out_data", metavar="OUT_DATA", help="Répertoire de sortie (listes, wav)."
    )
    pr.add_argument("--num-jobs", type=int, default=20, dest="num_jobs")
    pr.add_argument(
        "--downsampling",
        type=int,
        default=16000,
        dest="sampling_frequency",
        help="Fréquence d'échantillonnage cible (default: 16000).",
    )
    pr.add_argument(
        "--vad",
        action="store_true",
        default=False,
        help="Applique Silero VAD (VoxCeleb1/2 uniquement).",
    )
    pr.add_argument(
        "--delete-archives",
        action="store_true",
        default=False,
        dest="delete_archives",
        help="Supprime les archives après préparation.",
    )

    # --- Sous-commande : list ---
    sub.add_parser("list", help="Affiche les corpus disponibles.")

    return parser


def prepare_dataset(
    name: str,
    in_data: str,
    out_data: str,
    sampling_frequency: int = 16000,
    vad: bool = False,
    delete_archives: bool = False,
    num_jobs: int = 20,
) -> None:
    """Programmatic preparation entry point."""
    from .pipeline import PreparePipeline

    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset: '{name}'. Available: {list(REGISTRY)}")
    spec = REGISTRY[name]
    if spec.prepare_spec is None:
        raise ValueError(f"No PrepareSpec defined for '{name}'.")
    PreparePipeline().run(
        spec=spec,
        in_data=Path(in_data),
        out_data=Path(out_data),
        sampling_frequency=sampling_frequency,
        vad=vad,
        delete_archives=delete_archives,
        num_jobs=num_jobs,
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list" or (hasattr(args, "list") and args.list):
        rows = []
        for name, spec in REGISTRY.items():
            rows.append(
                (
                    name,
                    "✓" if spec.can_download else "—",
                    "✓" if spec.can_prepare else "—",
                    spec.description,
                )
            )

        name_w = max(len("dataset"), max(len(name) for name, _, _, _ in rows))
        dl_w = len("download")
        pr_w = len("prepare")
        description_w = len("description")

        print("Corpus disponibles :")
        print(
            f"{'dataset':<{name_w}}  {'download':<{dl_w}}  {'prepare':<{pr_w}} {'description':<{description_w}}"
        )
        print(f"{'-' * name_w}  {'-' * dl_w}  {'-' * pr_w} {'-' * description_w}")

        for name, download, prepare, description in rows:
            print(
                f"{name:<{name_w}}  {download:<{dl_w}}  {prepare:<{pr_w}} {description:<{description_w}}"
            )

        sys.exit(0)

    if args.command == "download":
        if not args.dataset or not args.target_dir:
            parser.parse_args(["download", "--help"])
            sys.exit(1)
        download_dataset(
            name=args.dataset,
            target_dir=args.target_dir,
            force=args.force,
            verify=args.verify,
            num_jobs=args.num_jobs,
        )

    elif args.command == "prepare":
        prepare_dataset(
            name=args.dataset,
            in_data=args.in_data,
            out_data=args.out_data,
            sampling_frequency=args.sampling_frequency,
            vad=args.vad,
            delete_archives=args.delete_archives,
            num_jobs=args.num_jobs,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
