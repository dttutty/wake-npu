"""Offline English/Chinese translation using local Argos models (CPU backend)."""

import argparse
import os
from pathlib import Path
import sys


DATA_ROOT = Path(__file__).resolve().parent / "translation-data"
for name, subdir in (
    ("XDG_DATA_HOME", "share"),
    ("XDG_CONFIG_HOME", "config"),
    ("XDG_CACHE_HOME", "cache"),
):
    os.environ.setdefault(name, str(DATA_ROOT / subdir))
os.environ.setdefault("ARGOS_PACKAGES_DIR", str(DATA_ROOT / "packages"))
os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"

from argostranslate import package, translate  # noqa: E402
from minisbd import models as sbd_models  # noqa: E402


LANGUAGE_PAIRS = (("zh", "en"), ("en", "zh"))


def install_models() -> None:
    package.update_package_index()
    available = package.get_available_packages()
    installed = {(p.from_code, p.to_code) for p in package.get_installed_packages()}
    for source, target in LANGUAGE_PAIRS:
        if (source, target) in installed:
            print(f"Already installed: {source} -> {target}")
            continue
        model = next(
            (p for p in available if (p.from_code, p.to_code) == (source, target)),
            None,
        )
        if model is None:
            raise RuntimeError(f"No model in Argos index for {source} -> {target}")
        print(f"Installing {source} -> {target}: {model}", flush=True)
        model.install()
    for language in ("zh-hans", "en"):
        print(f"Installing sentence boundary model: {language}", flush=True)
        sbd_models.get_model_file(language)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Text to translate; stdin if omitted")
    parser.add_argument("--install", action="store_true", help="Download both language models")
    parser.add_argument("--from", dest="source", choices=("zh", "en"), default="zh")
    parser.add_argument("--to", dest="target", choices=("zh", "en"), default="en")
    args = parser.parse_args()

    if args.install:
        install_models()
        return 0
    if args.source == args.target:
        parser.error("Source and target languages must differ")

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("Provide text as an argument or via stdin")

    installed = {(p.from_code, p.to_code) for p in package.get_installed_packages()}
    if (args.source, args.target) not in installed:
        parser.error("Language model is missing; run with --install first")

    print("Backend: CPU (CTranslate2); current SDK has no supported PHX/HPT translation path",
          file=sys.stderr)
    print(translate.translate(text, args.source, args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
