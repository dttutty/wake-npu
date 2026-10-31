"""Create a small INT8 QDQ convolution model for NPU smoke testing."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tiny_qdq_cnn.onnx"


def scalar(name: str, value: float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def main() -> None:
    rng = np.random.default_rng(8845)
    weights = rng.integers(-32, 33, size=(16, 3, 3, 3), dtype=np.int8)

    initializers = [
        scalar("input_scale", 1.0 / 128.0, np.float32),
        scalar("input_zero", 128, np.uint8),
        scalar("weight_scale", 1.0 / 64.0, np.float32),
        scalar("weight_zero", 0, np.int8),
        scalar("activation_scale", 1.0 / 64.0, np.float32),
        scalar("activation_zero", 0, np.uint8),
        numpy_helper.from_array(weights, name="weights_q"),
    ]

    nodes = [
        helper.make_node("QuantizeLinear", ["input", "input_scale", "input_zero"], ["input_q"]),
        helper.make_node("DequantizeLinear", ["input_q", "input_scale", "input_zero"], ["input_dq"]),
        helper.make_node("DequantizeLinear", ["weights_q", "weight_scale", "weight_zero"], ["weights_dq"]),
        helper.make_node("Conv", ["input_dq", "weights_dq"], ["conv"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("QuantizeLinear", ["relu", "activation_scale", "activation_zero"], ["relu_q"]),
        helper.make_node("DequantizeLinear", ["relu_q", "activation_scale", "activation_zero"], ["relu_dq"]),
        helper.make_node("GlobalAveragePool", ["relu_dq"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["output"], axis=1),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_qdq_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="amd-npu-smoke-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, MODEL_PATH)
    print(f"Created {MODEL_PATH}")


if __name__ == "__main__":
    main()
