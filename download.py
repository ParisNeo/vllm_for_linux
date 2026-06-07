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
    ASCIIColors.magenta(" By ParisNeo")
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
        ASCIIColors.yellow("Public downloads still work, but authenticated access is recommended")
        ASCIIColors.yellow("for higher rate limits, more reliable downloads, and gated model access.")
        ASCIIColors.white("")
        ASCIIColors.magenta("How to create and use a token:")
        ASCIIColors.white("1. Open: https://huggingface.co/settings/tokens")
        ASCIIColors.white("2. Create a new token with read access.")
        ASCIIColors.white("3. Export it in your shell:")
        ASCIIColors.white('   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"')
        ASCIIColors.white("4. Or login once with:")
        ASCIIColors.white("   hf auth login")
        ASCIIColors.white("")
        ASCIIColors.yellow("Tip: HF_TOKEN overrides the cached login token.")

    ASCIIColors.white("")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face repo snapshot into a local directory."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Hugging Face repo id, e.g. mistralai/Mistral-7B-Instruct-v0.2",
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

    base_dir = Path(args.dir).expanduser().resolve()
    target = base_dir / sanitize_repo_id(args.model)
    target.mkdir(parents=True, exist_ok=True)

    print_header(args.model, target)

    _, token = find_hf_token()

    try:
        if args.dry_run:
            ASCIIColors.blue("Dry run enabled. Querying remote snapshot without downloading...")
            dry_info = snapshot_download(
                repo_id=args.model,
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=str(target),
                local_dir_use_symlinks=False,
                allow_patterns=args.allow_pattern,
                ignore_patterns=args.ignore_pattern,
                token=token,
                dry_run=True,
            )
            ASCIIColors.green("Dry run completed.")
            ASCIIColors.white(str(dry_info))
            return

        ASCIIColors.blue("Starting snapshot download...")
        snapshot_path = snapshot_download(
            repo_id=args.model,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
            token=token,
            resume_download=True,
        )

        ASCIIColors.green("")
        ASCIIColors.green("Download completed successfully.")
        ASCIIColors.green(f"Snapshot path: {snapshot_path}")

    except KeyboardInterrupt:
        ASCIIColors.red("\nDownload interrupted by user.")
        sys.exit(130)
    except Exception as e:
        ASCIIColors.red("Download failed.")
        ASCIIColors.red(str(e))
        ASCIIColors.white("")
        ASCIIColors.yellow("Common fixes:")
        ASCIIColors.white("- Check that the repo id is correct.")
        ASCIIColors.white("- If the model is gated/private, ensure your HF token has access.")
        ASCIIColors.white("- Accept the model license on Hugging Face if required.")
        ASCIIColors.white("- Try setting HF_TOKEN or running: hf auth login")
        sys.exit(1)


if __name__ == "__main__":
    main()
