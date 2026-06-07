from pathlib import Path
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dir", type=str, default="models")
    args = parser.parse_args()

    target = Path(args.dir).resolve() / args.model.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model,
        local_dir=str(target),
        local_dir_use_symlinks=False
    )

if __name__ == "__main__":
    main()
