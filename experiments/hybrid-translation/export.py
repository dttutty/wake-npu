"""Export the installed Argos model to bounded FP32 encoder/decoder graphs."""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "iron-real-weight"))
from ct2_weights import ModelWeights

PACKAGE = ROOT / "translation-data/packages/translate-zh_en-1_9"
ARTIFACTS = Path(__file__).parent / "artifacts"
LENGTH, DIM, HEADS = 32, 512, 8


def configure(length):
    global LENGTH, ARTIFACTS
    if length not in (32, 64):
        raise ValueError("Supported sequence capacities are 32 and 64")
    LENGTH = length
    ARTIFACTS = Path(__file__).parent / "artifacts"
    if length != 32:
        ARTIFACTS = ARTIFACTS / f"length{length}"


def dequantized(model, name):
    name = model.aliases.get(name, name)
    values, scales = model.array(name), model.array(name + "_scale")
    if values.dtype != np.int8 or scales.shape != values.shape[:1] or (scales <= 0).any():
        raise ValueError(f"Unexpected quantization: {name}")
    return values.astype(np.float32) / scales[:, None]


class Graph:
    def __init__(self, model):
        self.model, self.nodes, self.initializers, self.inputs, self.outputs = model, [], {}, [], []

    def const(self, name, values):
        if name not in self.initializers:
            self.initializers[name] = nh.from_array(np.asarray(values), name)
        return name

    def op(self, kind, *args, **attrs):
        name = f"v{len(self.nodes)}"
        self.nodes.append(h.make_node(kind, list(args), [name], name=name, **attrs))
        return name

    def input(self, name, shape, dtype=T.FLOAT):
        self.inputs.append(h.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, value, name, shape, dtype=T.FLOAT):
        self.nodes.append(h.make_node("Identity", [value], [name], name=name))
        self.outputs.append(h.make_tensor_value_info(name, dtype, shape))

    def reshape(self, x, shape):
        return self.op("Reshape", x, self.const(f"shape{tuple(shape)}", np.array(shape, np.int64)))

    def slice(self, x, start, end):
        return self.op("Slice", x, self.const(f"start{start}", np.array([start], np.int64)),
                       self.const(f"end{end}", np.array([end], np.int64)),
                       self.const("last_axis", np.array([-1], np.int64)))

    def linear(self, x, prefix):
        w = self.const(prefix + "/w", dequantized(self.model, prefix + "/weight").T.copy())
        b = self.const(prefix + "/b", self.model.array(prefix + "/bias").astype(np.float32))
        return self.op("Add", self.op("MatMul", x, w), b)

    def norm(self, x, residual, prefix):
        gamma = self.const(prefix + "/g", self.model.array(prefix + "/gamma").astype(np.float32))
        beta = self.const(prefix + "/b", self.model.array(prefix + "/beta").astype(np.float32))
        return self.op("LayerNormalization", self.op("Add", x, residual), gamma, beta, axis=-1, epsilon=1e-5)

    def attention(self, q, k, v, mask, rows):
        q = self.op("Transpose", self.reshape(q, [rows, HEADS, 64]), perm=[1, 0, 2])
        k = self.op("Transpose", self.reshape(k, [LENGTH, HEADS, 64]), perm=[1, 2, 0])
        v = self.op("Transpose", self.reshape(v, [LENGTH, HEADS, 64]), perm=[1, 0, 2])
        q = self.op("Mul", q, self.const("qscale", np.array(0.125, np.float32)))
        scores = self.op("Add", self.op("MatMul", q, k), mask)
        context = self.op("MatMul", self.op("Softmax", scores, axis=-1), v)
        return self.reshape(self.op("Transpose", context, perm=[1, 0, 2]), [rows, DIM])

    def ffn(self, x, prefix):
        up = self.linear(x, prefix + "/linear_0")
        activated = self.op("Mul", up, self.op("Sigmoid", up))
        return self.norm(self.linear(activated, prefix + "/linear_1"), x, prefix + "/layer_norm")

    def save(self, name):
        graph = h.make_graph(self.nodes, name, self.inputs, self.outputs, list(self.initializers.values()))
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9)
        onnx.checker.check_model(model)
        onnx.save(model, ARTIFACTS / f"{name}.onnx")


def export():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = ModelWeights(PACKAGE / "model/model.bin")
    if model.model_type != "TransformerSpec" or model.revision != 7:
        raise ValueError("Expected Argos TransformerSpec revision 7")
    for scope in ("encoder", "decoder"):
        for key, value in (("num_heads", 8), ("pre_norm", 0), ("activation", 2), ("scale_embeddings", 1)):
            if model.array(f"{scope}/{key}").item() != value:
                raise ValueError(f"Unsupported {scope}/{key}")
    if model.array("decoder/start_from_zero_embedding").item() != 1:
        raise ValueError("Expected zero start embedding")
    g = Graph(model)
    x = g.input("source", [LENGTH, DIM])
    mask = g.input("source_mask", [1, LENGTH])
    for i in range(6):
        p = f"encoder/layer_{i}"
        qkv = g.linear(x, p + "/self_attention/linear_0")
        context = g.attention(*(g.slice(qkv, j * DIM, (j + 1) * DIM) for j in range(3)), mask, LENGTH)
        x = g.norm(g.linear(context, p + "/self_attention/linear_1"), x, p + "/self_attention/layer_norm")
        x = g.ffn(x, p + "/ffn")
    g.output(x, "memory", [LENGTH, DIM])
    g.save("encoder")

    g = Graph(model)
    memory = g.input("memory", [LENGTH, DIM])
    for i in range(6):
        kv = g.linear(memory, f"decoder/layer_{i}/attention/linear_1")
        for j, kind in enumerate(("k", "v")):
            g.output(g.slice(kv, j * DIM, (j + 1) * DIM), f"cross_{kind}_{i}", [LENGTH, DIM])
    g.save("cross")

    g = Graph(model)
    token = g.input("token", [1], T.INT64)
    step = g.input("step", [1], T.INT64)
    start_scale = g.input("embedding_scale", [1, 1])
    write = g.input("write_mask", [LENGTH, 1])
    self_mask = g.input("self_mask", [1, LENGTH])
    source_mask = g.input("source_mask", [1, LENGTH])
    embedding = g.const("embedding", dequantized(model, "decoder/projection/weight"))
    positions = g.const("positions", model.array("decoder/position_encodings/encodings").astype(np.float32))
    x = g.op("Add", g.op("Mul", g.op("Gather", embedding, token), start_scale), g.op("Gather", positions, step))
    keep = g.op("Sub", g.const("one", np.array(1, np.float32)), write)
    for i in range(6):
        p = f"decoder/layer_{i}"
        qkv = g.linear(x, p + "/self_attention/linear_0")
        q = g.slice(qkv, 0, DIM)
        cache = []
        for j, kind in enumerate(("k", "v"), 1):
            past = g.input(f"past_{kind}_{i}", [LENGTH, DIM])
            current = g.slice(qkv, j * DIM, (j + 1) * DIM)
            present = g.op("Add", g.op("Mul", past, keep), g.op("Mul", current, write))
            g.output(present, f"present_{kind}_{i}", [LENGTH, DIM])
            cache.append(present)
        context = g.attention(q, *cache, self_mask, 1)
        x = g.norm(g.linear(context, p + "/self_attention/linear_1"), x, p + "/self_attention/layer_norm")
        q = g.linear(x, p + "/attention/linear_0")
        cross = [g.input(f"cross_{kind}_{i}", [LENGTH, DIM]) for kind in ("k", "v")]
        context = g.attention(q, *cross, source_mask, 1)
        x = g.norm(g.linear(context, p + "/attention/linear_2"), x, p + "/attention/layer_norm")
        x = g.ffn(x, p + "/ffn")
    logits = g.op("Add", g.op("MatMul", x, g.op("Transpose", embedding, perm=[1, 0])),
                  g.const("projection_bias", model.array("decoder/projection/bias").astype(np.float32)))
    g.output(g.op("ArgMax", logits, axis=1, keepdims=0), "next_token", [1], T.INT64)
    g.output(x, "hidden", [1, DIM])
    g.save("decoder")
    print("Exported encoder, cross-K/V preparation, and cached decoder", flush=True)


if __name__ == "__main__":
    export()
