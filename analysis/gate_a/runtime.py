"""Offline loading and source-level production metric access for Gate A."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib
import json
import random
import sys
import types
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8388608), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_fob(implementation, width, checkpoint, canonical_unused=False):
    """Skip redundant backbone bootstrap ONLY; full FoB state loads strictly.
    The bootstrap adapter never alters forward, refinement, ordering or weights
    after strict loading. It prevents placeholder file access / URL downloads.
    """
    import torch
    label = f"_gate_a_{implementation}_{width}"
    package = types.ModuleType(label)
    relative = "FoB_SAM_upstream" if implementation == "upstream" else "FoB_SAM"
    package.__path__ = [str(ROOT / "third_party" / relative / "models")]
    sys.modules[label] = package
    # Inference never invokes this unavailable training-only loss.
    if "info_nce" not in sys.modules and importlib.util.find_spec("info_nce") is None:
        loss_module = types.ModuleType("info_nce")
        class DisabledTrainingLoss:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                raise RuntimeError("Training loss called in inference-only audit")
        loss_module.InfoNCE = DisabledTrainingLoss
        sys.modules["info_nce"] = loss_module
    enc = importlib.import_module(label + ".encoder")
    cls = importlib.import_module(label + ".FoB").FewShotSeg
    original = enc.Res101Encoder.__init__
    def offline_init(self, replace_stride_with_dilation=None, pretrained_weights=None):
        original(self, replace_stride_with_dilation=replace_stride_with_dilation, pretrained_weights=None)
    print(f"construct {implementation} width={width}; offline redundant-bootstrap bypass", flush=True)
    with patch.object(enc.Res101Encoder, "__init__", offline_init):
        model = cls(types.SimpleNamespace(dataset="SABS", max_points=width, n_ways=1, n_shots=1)).cuda().eval()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
    before = {k: list(v.shape) for k, v in state.items()}
    raw_error = None
    try:
        result = model.load_state_dict(copy.deepcopy(state), strict=True)
    except RuntimeError as exc:
        raw_error = str(exc)
        missing = set(model.state_dict()) - set(state)
        extra = set(state) - set(model.state_dict())
        if not canonical_unused or missing != {"encoder.reduce2.weight"} or extra:
            raise
        forward_source = (ROOT / "third_party" / relative / "models/encoder.py").read_text()
        forward = next(n for c in ast.parse(forward_source).body if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        if any(isinstance(n, ast.Attribute) and n.attr == "reduce2" for n in ast.walk(forward)):
            raise RuntimeError("Cannot canonicalize a used parameter")
        del model.encoder.reduce2
        result = model.load_state_dict(copy.deepcopy(state), strict=True)
    if implementation == "local":
        model.allocator = None
    after = {k: list(v.shape) for k, v in model.state_dict().items()}
    differences = {k: {"checkpoint": shape, "model": after.get(k)} for k, shape in before.items() if shape != after.get(k)}
    metadata = {"implementation": implementation, "width": width, "raw_strict_load_error": raw_error,
                "schema_variant": "unused_reduce2_removed" if raw_error else "raw",
                "training_loss_adapter": "guarded unavailable InfoNCE; no inference calls", "checkpoint_sha256": sha256(checkpoint),
                "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
                "shape_adaptations": differences,
                "bootstrap": "pretrained_weights=None only during construction; full checkpoint strict=True",
                "all_model_state_keys_loaded": set(after) == set(before)}
    print(json.dumps(metadata), flush=True)
    return model, metadata


def official_sam_b(checkpoint):
    import torch
    sys.path.insert(0, str(ROOT / "third_party/segment-anything"))
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = sam.load_state_dict(state, strict=True)
    print("SAM-B loaded strictly", result, flush=True)
    return SamPredictor(sam.cuda().eval()), {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys, "sha256": sha256(checkpoint)}


def production_metrics():
    """Execute the ACTUAL production function AST, avoiding module auto-installs."""
    from medpy.metric.binary import hd95
    source = ROOT / "experiments/run_step4_baselines.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_metrics")
    namespace = {"np": np, "medpy_hd95": hd95}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["compute_metrics"], namespace
