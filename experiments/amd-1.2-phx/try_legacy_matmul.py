import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent


def make_model(width, mode):
    rng = np.random.default_rng(2026)
    if mode == "int8":
        weight = rng.integers(-3, 4, (width, width), dtype=np.int8)
        x = rng.integers(0, 7, (1, width), dtype=np.uint8)
        node = helper.make_node("MatMulInteger", ["x", "weight", "x_zp", "w_zp"], ["y"])
        initializers = [numpy_helper.from_array(weight, "weight"),
                        numpy_helper.from_array(np.array(0, dtype=np.uint8), "x_zp"),
                        numpy_helper.from_array(np.array(0, dtype=np.int8), "w_zp")]
        input_type, output_type = TensorProto.UINT8, TensorProto.INT32
        imports = [helper.make_opsetid("", 18)]
    else:
        block_size = 32
        if width % block_size:
            raise ValueError("width must be divisible by 32")
        weight = rng.integers(0, 256, (width, width // block_size, block_size // 2), dtype=np.uint8)
        scales = np.full(width * (width // block_size), 0.125, dtype=np.float32)
        x = rng.uniform(-1, 1, (1, width)).astype(np.float32)
        node = helper.make_node("MatMulNBits", ["x", "weight", "scales"], ["y"],
                                domain="com.microsoft", K=width, N=width,
                                bits=4, block_size=block_size)
        initializers = [numpy_helper.from_array(weight, "weight"),
                        numpy_helper.from_array(scales, "scales")]
        input_type = output_type = TensorProto.FLOAT
        imports = [helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
    graph = helper.make_graph(
        [node],
        "legacy_phx_matmul",
        [helper.make_tensor_value_info("x", input_type, [1, width])],
        [helper.make_tensor_value_info("y", output_type, [1, width])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=imports)
    model.ir_version = 10
    return model, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--mode", choices=["int8", "int4"], default="int4")
    parser.add_argument("--unsafe-npu", action="store_true",
                        help="Run the legacy EP; it may crash the Python process")
    args = parser.parse_args()

    os.environ["DEVICE"] = "phx"
    os.environ["XLNX_VART_FIRMWARE"] = str(ROOT / "1x4.xclbin")
    os.environ["PYTORCH_AIE_PATH"] = str(ROOT)
    os.environ["XLNX_CACHE_DIR"] = str(ROOT / "cache")
    model, x = make_model(args.width, args.mode)
    model_path = ROOT / "cache" / f"{args.mode}_{args.width}.onnx"
    model_path.parent.mkdir(exist_ok=True)
    onnx.save(model, model_path)
    cpu = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    expected = cpu.run(None, {"x": x})[0]
    print("CPU checksum:", float(expected.sum()), "nonzero:", int(np.count_nonzero(expected)), flush=True)
    if not args.unsafe_npu:
        print("NPU skipped: pass --unsafe-npu only if you accept a native Python crash.", flush=True)
        return

    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0002)

    options = ort.SessionOptions()
    options.log_severity_level = 2
    options.enable_profiling = True
    ep = ort.InferenceSession(
        str(model_path), options,
        providers=[("VitisAIExecutionProvider", {"config_file": str(ROOT / "vaip_config_transformers.json")})],
    )
    print("Providers:", ep.get_providers(), flush=True)
    actual = ep.run(None, {"x": x})[0]
    print("EP checksum:", float(actual.sum()), "nonzero:", int(np.count_nonzero(actual)), flush=True)
    print("Matches CPU:", np.allclose(actual, expected, rtol=0.05, atol=0.05), flush=True)
    print("Profile:", ep.end_profiling(), flush=True)


if __name__ == "__main__":
    main()
