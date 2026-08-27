#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys

from huggingface_hub import snapshot_download
from ascii_colors import ASCIIColors


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def find_hf_token():
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env_token:
        return ("environment", env_token)

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_file = hf_home / "token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return ("cache", token)

    return (None, None)


def print_header(model: str, target: Path):
    ASCIIColors.cyan("=" * 80)
    ASCIIColors.cyan(" Hugging Face model snapshot downloader")
    ASCIIColors.magenta(" By ParisNeo & Optimized for vLLM Cluster")
    ASCIIColors.cyan("=" * 80)
    ASCIIColors.white(f"Model  : {model}")
    ASCIIColors.white(f"Target : {target}")
    ASCIIColors.white("")

    token_source, _ = find_hf_token()

    if token_source:
        ASCIIColors.green("HF token detected.")
        ASCIIColors.green(f"Auth source: {token_source}")
    else:
        ASCIIColors.yellow("No Hugging Face token detected.")
        ASCIIColors.yellow("Public downloads still work, but authenticated access is recommended.")
    ASCIIColors.white("")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face repo snapshot into a local directory."
    )
    parser.add_argument(
        "positional_model",
        type=str,
        nargs="?",
        default="unsloth/GLM-5.3-Flash-FP8",
        help="Hugging Face repo id passed directly (default: unsloth/GLM-5.3-Flash-FP8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Alternative way to provide Hugging Face repo id",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="models",
        help="Base directory where the model snapshot will be stored",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch, tag, or commit revision",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional file pattern to include, can be repeated",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional file pattern to exclude, can be repeated",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query what would be downloaded without downloading files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_id = args.positional_model or args.model
    if not model_id:
        ASCIIColors.red("Error: You must provide a model ID.")
        sys.exit(1)

    if model_id == "unsloth/GLM-5.3-Flash-FP8":
        ASCIIColors.cyan("No model specified. Defaulting to unsloth/GLM-5.3-Flash-FP8.")

    script_dir = Path(__file__).parent.resolve()
    base_dir = (script_dir / args.dir).resolve()
    target = base_dir / sanitize_repo_id(model_id)
    target.mkdir(parents=True, exist_ok=True)

    print_header(model_id, target)

    _, token = find_hf_token()

    ignore_patterns = args.ignore_pattern or []
    if not args.allow_pattern:
        ignore_patterns.extend(["*.pth", "*.pt", "*.bin"])

    try:
        if args.dry_run:
            ASCIIColors.blue("Dry run enabled. Querying remote snapshot without downloading...")
            dry_info = snapshot_download(
                repo_id=model_id,
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=str(target),
                local_dir_use_symlinks=False,
                allow_patterns=args.allow_pattern,
                ignore_patterns=ignore_patterns,
                token=token,
                dry_run=True,
            )
            ASCIIColors.green("Dry run completed.")
            ASCIIColors.white(str(dry_info))
            return

        ASCIIColors.blue("Starting snapshot download...")
        snapshot_path = snapshot_download(
            repo_id=model_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            allow_patterns=args.allow_pattern,
            ignore_patterns=ignore_patterns,
            token=token,
            resume_download=True,
            max_workers=8,
        )

        ASCIIColors.green("")
        ASCIIColors.green("Download completed successfully.")
        ASCIIColors.green(f"Snapshot path: {snapshot_path}")

    except KeyboardInterrupt:
        ASCIIColors.red("\nDownload interrupted by user.")
        sys.exit(130)
    except Exception as e:
        ASCIIColors.red(f"Download failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()