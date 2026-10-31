"""Decode target SentencePiece pieces in the installed Argos environment."""

import json
import sys
from pathlib import Path

from argostranslate.tokenizer import SentencePieceTokenizer


tokenizer = SentencePieceTokenizer(Path(sys.argv[1]))
pieces = json.load(sys.stdin)
text = tokenizer.decode(pieces)
print(text[1:] if text.startswith(" ") else text)
