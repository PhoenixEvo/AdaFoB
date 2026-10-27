"""Create new named drivers from preserved Step8 execution primitives."""
from pathlib import Path
p=Path('analysis/controls_v1')
s=Path('analysis/rebaseline_v1/inference.py').read_text()
s=s.replace('"""Resumable corrected three-arm full run. One embedding, three decoder calls."""','"""Original-pool controls: exclusive slice commits and shared image embedding."""')
s=s.replace('from .common import *','from analysis.rebaseline_v1.common import *\nfrom .core import ARMS,run_slice')
s=s.replace('import torch\n    seed_all','baseline_expected=json.loads((a.lock_dir/"step8_fingerprints.json").read_text())\n    import torch\n    seed_all')
s=s.replace('arrays,record,_=run_shared(model,predictor,case)','try:\n                    arrays,record=run_slice(model,predictor,case,c,baseline_expected[f"{case[\'episode_id\']}:{case[\'native_z\']}"])\n                except Exception as exc:\n                    write_json(folder/f"failure_{time.time_ns()}.json",{"episode_id":case["episode_id"],"native_z":case["native_z"],"error":repr(exc)})\n                    raise')
with (p/'inference.py').open('x') as f:f.write(s)
s=Path('analysis/rebaseline_v1/remote.py').read_text().replace('from .common import','from analysis.rebaseline_v1.common import')
s=s.replace('analysis.rebaseline_v1.pilot','analysis.controls_v1.pilot').replace('analysis.rebaseline_v1.inference','analysis.controls_v1.inference')
s=s.replace('step8_sam_b','step9_sam_b').replace('"new_arms":3','"new_arms":31').replace('120*3*2','120*31*2').replace('"limit_gpu_hours":2.5','"limit_gpu_hours":3.0')
with (p/'remote.py').open('x') as f:f.write(s)
print('New execution drivers created; Step8 source unchanged',flush=True)