"""Compare CPU, DirectML GPU, and IRON NPU encoder + DirectML decoder."""

import argparse
import json
import os
import subprocess
from collections import Counter
from graphlib import TopologicalSorter
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

import export as graph_export
from export import ARTIFACTS, DIM, LENGTH, PACKAGE, ROOT, ModelWeights, dequantized, export


class Backend:
    def __init__(self, device, suffix=""):
        self.device, self.sessions = device, {}
        started = perf_counter()
        for name in ("encoder", "cross", "decoder"):
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 4
            options.enable_profiling = True
            options.profile_file_prefix = str(ARTIFACTS / f"profile_{device}{suffix}_{name}")
            if device == "dml":
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            providers = [("DmlExecutionProvider", {"device_id": 0})] if device == "dml" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(ARTIFACTS / f"{name}{suffix}.onnx"), options, providers=providers)
            session.disable_fallback()
            if device == "dml" and "DmlExecutionProvider" not in session.get_providers():
                raise RuntimeError("DirectML session was not created")
            self.sessions[name] = session
        self.setup_s = perf_counter() - started
        print(f"Sessions {device}{suffix}: {self.setup_s:.3f}s", flush=True)

    def tensor(self, x):
        return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(x), self.device, 0)

    def run(self, name, inputs):
        session = self.sessions[name]
        binding = session.io_binding()
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                binding.bind_cpu_input(key, value)
            else:
                binding.bind_ortvalue_input(key, value)
        names = [item.name for item in session.get_outputs()]
        for key in names:
            binding.bind_output(key, self.device, 0)
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return dict(zip(names, binding.get_outputs()))

    def decode(self, memory, mask, eos, max_tokens):
        started = perf_counter()
        cross = self.run("cross", {"memory": memory})
        cache = {f"past_{kind}_{i}": self.tensor(np.zeros((LENGTH, DIM), np.float32))
                 for i in range(6) for kind in ("k", "v")}
        mask_gpu = self.tensor(mask)
        ids, states, token_times = [], [], []
        for step in range(max_tokens):
            tick = perf_counter()
            write = np.zeros((LENGTH, 1), np.float32)
            write[step] = 1
            self_mask = np.full((1, LENGTH), -10000, np.float32)
            self_mask[:, :step + 1] = 0
            inputs = dict(cross, **cache, source_mask=mask_gpu,
                          token=np.array([ids[-1] if ids else 0], np.int64),
                          step=np.array([step], np.int64), write_mask=write,
                          self_mask=self_mask,
                          embedding_scale=np.array([[np.sqrt(DIM) if step else 0]], np.float32))
            output = self.run("decoder", inputs)
            ids.append(int(output["next_token"].numpy()[0]))
            token_times.append(perf_counter() - tick)
            # Retain device buffers for numerical checks after the timed loop.
            states.append(output["hidden"])
            cache = {f"past_{kind}_{i}": output[f"present_{kind}_{i}"]
                     for i in range(6) for kind in ("k", "v")}
            if ids[-1] == eos:
                break
        elapsed = perf_counter() - started
        hidden = np.concatenate([state.numpy() for state in states])
        if not np.isfinite(hidden).all():
            raise AssertionError("Non-finite decoder hidden state")
        if self.device == "dml" and any(v.device_name() != "dml" for v in cache.values()):
            raise AssertionError("KV cache did not remain on GPU")
        return ids, hidden, elapsed, token_times

    def profiles(self):
        result = {}
        for name, session in self.sessions.items():
            path = session.end_profiling()
            events = json.loads(Path(path).read_text())
            counts = Counter(event.get("args", {}).get("provider") for event in events if event.get("args", {}).get("provider"))
            if self.device == "dml" and counts.get("CPUExecutionProvider", 0):
                raise AssertionError(f"Unexpected CPU graph execution: {counts}")
            if self.device == "dml" and not counts.get("DmlExecutionProvider", 0):
                raise AssertionError(f"No profiled DirectML execution for {name}")
            result[name] = {"path": path, "provider_events": dict(counts)}
        return result


class NpuBridge:
    def __init__(self):
        vcvars = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat")
        iron = ROOT / "third_party/mlir-aie/iron_env.cmd"
        worker = Path(__file__).with_name("npu_bridge.py")
        env = os.environ.copy()
        env["NPU_CACHE_HOME"] = str(ROOT / "tools/iron-cache")
        env["PYTHONIOENCODING"] = "utf-8"
        command = f'call "{vcvars}" >nul && call "{iron}" && python -u "{worker}"'
        started = perf_counter()
        self.process = subprocess.Popen(f'cmd.exe /d /s /c "{command}"', stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env)
        self.wait_line("READY")
        self.startup_s = perf_counter() - started

    def wait_line(self, prefix):
        for line in self.process.stdout:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
            print("IRON: " + line.rstrip(), flush=True)
        raise RuntimeError(f"NPU worker exited: {self.process.wait(timeout=10)}")

    def encode(self, source, pieces):
        path = ARTIFACTS / "npu_memory.npy"
        source_path = ARTIFACTS / "npu_source.npy"
        started = perf_counter()
        np.save(source_path, source)
        self.process.stdin.write(json.dumps({"source": str(source_path), "pieces": pieces, "output": str(path)}) + "\n")
        self.process.stdin.flush()
        info = json.loads(self.wait_line("RESULT "))
        memory = np.load(path)
        info["handoff_wall_s"] = perf_counter() - started
        return memory, info

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


def tokenize(texts, max_tokens):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"),
                             str(Path(__file__).with_name("prepare_cases.py")), str(PACKAGE), str(max_tokens)],
                            input=json.dumps(texts), capture_output=True, text=True, encoding="utf-8", env=env, check=True)
    return json.loads(result.stdout)


def fp16_models():
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    for name in ("encoder", "cross", "decoder"):
        path = ARTIFACTS / f"{name}_fp16.onnx"
        if not path.exists() or path.stat().st_mtime < (ARTIFACTS / f"{name}.onnx").stat().st_mtime:
            model = convert_float_to_float16(onnx.load(ARTIFACTS / f"{name}.onnx"), keep_io_types=True)
            # The converter appends boundary Cast nodes after their consumers.
            nodes = list(model.graph.node)
            producers = {output: i for i, node in enumerate(nodes) for output in node.output}
            dependencies = {i: {producers[x] for x in node.input if x in producers}
                            for i, node in enumerate(nodes)}
            order = list(TopologicalSorter(dependencies).static_order())
            del model.graph.node[:]
            model.graph.node.extend(nodes[i] for i in order)
            onnx.checker.check_model(model)
            onnx.save(model, path)


def main(args):
    global LENGTH, ARTIFACTS
    graph_export.configure(args.length)
    LENGTH, ARTIFACTS = graph_export.LENGTH, graph_export.ARTIFACTS
    if args.max_tokens is None:
        args.max_tokens = LENGTH
    if args.repeats < 1 or not 1 <= args.max_tokens <= LENGTH:
        raise ValueError(f"Positive repeats and max_tokens in [1,{LENGTH}] required")
    if args.export or not all((ARTIFACTS / f"{name}.onnx").exists() for name in ("encoder", "cross", "decoder")):
        export()
    cases = tokenize(args.text or ["你好，世界。", "人工智能正在改变世界。", "谢谢你。"], args.max_tokens)
    vocabulary = json.loads((PACKAGE / "model/shared_vocabulary.json").read_text(encoding="utf-8"))
    ids = {piece: i for i, piece in enumerate(vocabulary)}
    model = ModelWeights(PACKAGE / "model/model.bin")
    embeddings = dequantized(model, "encoder/embeddings_0/weight")
    positions = model.array("encoder/position_encodings/encodings")
    for case in cases:
        if len(case["pieces"]) > LENGTH:
            raise ValueError(f"Source exceeds {LENGTH} pieces including EOS")
    backends = [("cpu_fp32", Backend("cpu")), ("gpu_fp32", Backend("dml"))]
    if args.fp16:
        fp16_models()
        backends.append(("gpu_fp16", Backend("dml", "_fp16")))
    report = {"onnxruntime": ort.__version__, "repeats": args.repeats, "gpu_device_id": 0,
              "capacity": LENGTH, "record_mismatches": args.record_mismatches,
              "timing_note": "Warm requests with profiling enabled; excludes shared tokenization/source embedding preparation, session/JIT startup, and correctness readbacks. Hybrid includes host/file handoff and GPU upload. CPU CT2 has one warmed sample, four threads, INT8, greedy decoding; ONNX modes use the median of repeats.",
              "sessions_s": {name: backend.setup_s for name, backend in backends}, "cases": []}
    bridge = None
    try:
        if not args.skip_npu:
            bridge = NpuBridge()
            report["npu_worker_startup_s"] = bridge.startup_s
        for case in cases:
            length = len(case["pieces"])
            source = np.zeros((LENGTH, DIM), np.float32)
            source[:length] = embeddings[[ids.get(p, ids["<unk>"]) for p in case["pieces"]]] * np.sqrt(DIM) + positions[:length]
            mask = np.full((1, LENGTH), -10000, np.float32)
            mask[:, :length] = 0
            entry = dict(case, source_tokens=length, results={})
            reference_ids = reference_hidden = reference_memory = None
            for name, backend in backends:
                modes = [name] + (["npu_" + name] if bridge and backend.device == "dml" else [])
                for mode in modes:
                    samples = []
                    for repeat in range(args.repeats + 1):
                        tick = perf_counter()
                        npu_info = None
                        if mode.startswith("npu_"):
                            memory_np, npu_info = bridge.encode(source, case["pieces"])
                            memory = backend.tensor(memory_np)
                        else:
                            memory = backend.run("encoder", {"source": backend.tensor(source),
                                                               "source_mask": backend.tensor(mask)})["memory"]
                        encoder_s = perf_counter() - tick
                        selected, hidden, decode_s, token_times = backend.decode(memory, mask, ids["</s>"], args.max_tokens)
                        total = encoder_s + decode_s
                        memory_np = memory.numpy()
                        if reference_ids is None:
                            reference_ids, reference_hidden, reference_memory = selected, hidden, memory_np
                        matched = selected == reference_ids
                        first_difference = next((i for i, (a, b) in enumerate(zip(selected, reference_ids)) if a != b),
                                                min(len(selected), len(reference_ids)) if not matched else None)
                        if not matched and not args.record_mismatches:
                            raise AssertionError(f"Token mismatch for {case['text']} on {mode}: {[vocabulary[i] for i in selected]}")
                        # Hidden states remain comparable only while the input token histories agree.
                        compared_steps = min(len(hidden), len(reference_hidden),
                                             first_difference + 1 if first_difference is not None else len(hidden))
                        relative = float(np.linalg.norm(hidden[:compared_steps] - reference_hidden[:compared_steps]) /
                                         np.linalg.norm(reference_hidden[:compared_steps]))
                        if relative > 0.05 and not args.record_mismatches:
                            raise AssertionError(f"Decoder relative L2 {relative:.6f} exceeds 5% on {mode}")
                        if repeat:
                            samples.append({"encoder_and_handoff_s": encoder_s, "decode_s": decode_s, "total_s": total,
                                            "token_s": token_times, "hidden_relative_l2": relative if matched else None,
                                            "shared_history_relative_l2": relative, "compared_steps": compared_steps,
                                            "matches_onnx_cpu": matched, "first_difference_step": first_difference,
                                            "numerical_check_passed": matched and relative <= 0.05,
                                            "encoder_relative_l2": float(np.linalg.norm(memory_np[:length] - reference_memory[:length]) / np.linalg.norm(reference_memory[:length])),
                                            "npu": npu_info})
                    pieces = [vocabulary[i] for i in selected]
                    match_ct2 = [p for p in pieces if p != "</s>"] == case["ct2_pieces"]
                    entry["results"][mode] = {"pieces": pieces, "eos": selected[-1] == ids["</s>"], "matches_ct2": match_ct2,
                                               "matches_onnx_cpu": all(s["matches_onnx_cpu"] for s in samples),
                                               "numerical_check_passed": all(s["numerical_check_passed"] for s in samples),
                                               "median_s": float(np.median([s["total_s"] for s in samples])), "samples": samples}
                    print(f"{case['text']} | {mode}: {entry['results'][mode]['median_s']:.3f}s, ONNX match={matched}, CT2 match={match_ct2}, EOS={selected[-1] == ids['</s>']}, tokens={pieces}", flush=True)
            report["cases"].append(entry)
        report["profiles"] = {name: backend.profiles() for name, backend in backends}
        report["numerical_check_passed"] = all(result["numerical_check_passed"] for case in report["cases"] for result in case["results"].values())
        path = args.report or ARTIFACTS / ("results_fp16.json" if args.fp16 else "results.json")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        status = "PASS" if report["numerical_check_passed"] else "DIAGNOSTIC COMPLETE: numerical equivalence FAILED"
        print(f"{status}; GPU KV residency and provider profiles checked. Report: {path}")
    finally:
        if bridge:
            bridge.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--length", type=int, choices=(32, 64), default=32)
    parser.add_argument("--record-mismatches", action="store_true", help="Record numerical failures and finish the diagnostic benchmark")
    parser.add_argument("--skip-npu", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report", type=Path, help="Override the JSON report path")
    main(parser.parse_args())
