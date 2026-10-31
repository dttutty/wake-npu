"""Chunked BF16 NPU projection for the Argos decoder vocabulary."""

import numpy as np
from ml_dtypes import bfloat16

from attention import DIM, mm, upload
from device_profile import call, readback
from vocab_top2 import CHUNKS, RESULT_FLOATS, bias_top2, merge_top2

import aie.iron as iron


QUERY_ROWS = 64
VOCAB_TILE = 1024


class NpuVocabHead:
    def __init__(self, projection, bias):
        if projection.shape != (65000, DIM) or bias.shape != (65000,):
            raise ValueError("Unexpected vocabulary projection shape")
        self.vocabulary_size = projection.shape[0]
        self.tiles = []
        for start in range(0, self.vocabulary_size, VOCAB_TILE):
            count = min(VOCAB_TILE, self.vocabulary_size - start)
            weight = np.zeros((DIM, VOCAB_TILE), dtype=bfloat16)
            weight[:, :count] = projection[start:start + count].T.astype(bfloat16)
            offsets = np.full(VOCAB_TILE, -1.0e30, dtype=np.float32)
            offsets[:count] = bias[start:start + count]
            self.tiles.append((start, count, upload(weight),
                               call("tensor upload", iron.tensor, offsets, dtype=np.float32, device="npu"), offsets))

    def top2(self, hidden, *, capture_logits=False):
        if hidden.shape != (DIM,):
            raise ValueError("Unexpected hidden state shape")
        query = np.zeros((QUERY_ROWS, DIM), dtype=bfloat16)
        query[0] = hidden.astype(bfloat16)
        query_npu = upload(query)
        captured = np.empty(self.vocabulary_size, dtype=np.float32) if capture_logits else None
        candidates_npu = iron.zeros((CHUNKS, RESULT_FLOATS), dtype=np.float32, device="npu")
        for tile, (start, count, weight_npu, bias_npu, offsets) in enumerate(self.tiles):
            output_npu = mm(query_npu, weight_npu, QUERY_ROWS, DIM, VOCAB_TILE, output_dtype=np.float32)
            top_npu = candidates_npu.subview(tile * RESULT_FLOATS * 4, (RESULT_FLOATS,))
            call("vocab bias top2", bias_top2, output_npu, bias_npu, top_npu)
            if captured is not None:
                raw = readback(output_npu).reshape(QUERY_ROWS, VOCAB_TILE)[0, :count]
                captured[start:start + count] = raw + offsets[:count]
        merged_npu = iron.zeros(4, dtype=np.float32, device="npu")
        call("vocab merge top2", merge_top2, candidates_npu, merged_npu)
        merged = readback(merged_npu)
        if not np.isfinite(merged).all():
            raise AssertionError("Non-finite NPU vocabulary top-2")
        token_id = int(merged[1])
        second_id = int(merged[3])
        if (not 0 <= token_id < self.vocabulary_size or
                not 0 <= second_id < self.vocabulary_size or second_id == token_id):
            raise AssertionError("NPU vocabulary head returned invalid top-2 IDs")
        if captured is not None:
            if not np.isfinite(captured).all():
                raise AssertionError("Non-finite NPU vocabulary logits")
            np.testing.assert_allclose(captured[token_id], merged[0], atol=1e-5)
            np.testing.assert_allclose(captured[second_id], merged[2], atol=1e-5)
            if int(np.argmax(captured)) != token_id:
                raise AssertionError("NPU top-2 disagrees with full vocabulary logits")
            np.testing.assert_allclose(np.partition(captured, -2)[-2], merged[2], atol=1e-5)
        return token_id, float(merged[0]), float(merged[2]), captured
