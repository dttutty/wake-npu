"""Use the existing Argos Python environment to segment source text."""

import json
import sys

import sentencepiece


model = sentencepiece.SentencePieceProcessor(model_file=sys.argv[1])
print(json.dumps(model.encode(sys.argv[2], out_type=str), ensure_ascii=False))
