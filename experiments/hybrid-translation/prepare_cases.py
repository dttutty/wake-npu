"""Use the installed Argos tokenizer and independent CTranslate2 greedy baseline."""

import json
import sys
from time import perf_counter

import ctranslate2
import sentencepiece

package = sys.argv[1]
max_tokens = int(sys.argv[2])
texts = json.load(sys.stdin)
tokenizer = sentencepiece.SentencePieceProcessor(model_file=package + "/sentencepiece.model")
translator = ctranslate2.Translator(package + "/model", device="cpu", compute_type="int8", inter_threads=1, intra_threads=4)
results = []
for text in texts:
    pieces = tokenizer.encode(text, out_type=str)
    source = pieces + ["</s>"]
    translator.translate_batch([source], beam_size=1, max_decoding_length=max_tokens)
    started = perf_counter()
    result = translator.translate_batch([source], beam_size=1, max_decoding_length=max_tokens)[0]
    results.append({"text": text, "pieces": source, "ct2_pieces": result.hypotheses[0],
                    "ct2_s": perf_counter() - started})
print(json.dumps(results, ensure_ascii=False))
