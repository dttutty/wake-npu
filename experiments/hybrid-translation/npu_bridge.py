"""Persistent IRON subprocess; one source-memory handoff per sentence."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "iron-real-weight"))
from run import ModelWeights, PACKAGE
from encoder_stack import encode

model = ModelWeights(PACKAGE / "model/model.bin")
print("READY", flush=True)
for line in sys.stdin:
    request = json.loads(line)
    started = perf_counter()
    source = np.zeros((64, 512), dtype=bfloat16)
    loaded = np.load(request["source"])
    if loaded.ndim != 2 or loaded.shape[1] != 512 or loaded.shape[0] not in (32, 64):
        raise ValueError("Expected a 32x512 or 64x512 source tensor")
    source[:loaded.shape[0]] = loaded.astype(bfloat16)
    pieces = request["pieces"]
    if not 1 <= len(pieces) <= loaded.shape[0]:
        raise ValueError("Source token count exceeds tensor capacity")
    prepared = perf_counter()
    memory, _ = encode(model, source, pieces, report=False, verify=False)
    encoded = perf_counter()
    np.save(request["output"], memory[:loaded.shape[0]].astype(np.float32))
    print("RESULT " + json.dumps({"prepare_s": prepared - started,
                                  "encoder_s": encoded - prepared,
                                  "worker_s": perf_counter() - started,
                                  "pieces": pieces}), flush=True)
