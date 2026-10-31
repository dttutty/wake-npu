import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def words(text):
    text = text.lower().replace("i'm", "i am")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def edit_counts(reference, hypothesis):
    rows = [[(0, 0, 0, 0)] * (len(hypothesis) + 1)]
    for i in range(1, len(reference) + 1):
        rows.append([(i, 0, i, 0)] + [(0, 0, 0, 0)] * len(hypothesis))
    for j in range(1, len(hypothesis) + 1):
        rows[0][j] = (j, 0, 0, j)
    for i, ref in enumerate(reference, 1):
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                rows[i][j] = rows[i - 1][j - 1]
                continue
            substitute = rows[i - 1][j - 1]
            delete = rows[i - 1][j]
            insert = rows[i][j - 1]
            rows[i][j] = min(
                (substitute[0] + 1, substitute[1] + 1, substitute[2], substitute[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
    return rows[-1][-1]


def main():
    samples = json.loads((ROOT / "asr-comparison.json").read_text(encoding="utf-8"))
    transcripts = json.loads((ROOT / "whisper-transcripts.json").read_text(encoding="utf-8"))
    totals = {engine: [0, 0] for engine in ("small.en", "Live Captions")}
    print("sample                  words  small.en  Live Captions")
    for sample in samples:
        reference = words(sample["reference"])
        whisper = transcripts[sample["whisper_file"]]
        errors = {}
        for engine, transcript in (("small.en", whisper), ("Live Captions", sample["live_captions"])):
            distance, substitutions, deletions, insertions = edit_counts(reference, words(transcript))
            errors[engine] = distance
            totals[engine][0] += distance
            totals[engine][1] += len(reference)
        print(f"{sample['sample']:<23} {len(reference):>5} {errors['small.en']:>9} {errors['Live Captions']:>14}")
    for engine, (errors, total_words) in totals.items():
        print(f"{engine}: {errors}/{total_words} = {errors / total_words:.2%} WER")


if __name__ == "__main__":
    main()
