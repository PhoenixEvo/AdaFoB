"""CPU-only input parity and resumability tests; no local CUDA invocation."""
import importlib.util,json,sys,types,os
from pathlib import Path
import numpy as np,pandas as pd
import pytest
from analysis.rebaseline_v1.common import episode_inputs,read_config,read_manifest,serialized_slice,configurations
from analysis.rebaseline_v1.inference import save_slice,load_completed

@pytest.fixture
def config():
    return {"support_parts":3,"support_patients":{"0":"image_2","1":"image_8","2":"image_14","3":"image_20","4":"image_25"}}

def test_exact_inputs_against_actual_benchmark_loader(config):
    # CPU only: no seed_all or CUDA calls.
    data_dir=Path(os.environ["ADAF_OB_DATA_DIR"])
    root=Path.cwd();package=types.ModuleType("_rebaseline_cpu_data")
    package.__path__=[str(root/"third_party/FoB_SAM/dataloaders")];sys.modules[package.__name__]=package
    import importlib
    Dataset=importlib.import_module(package.__name__+".datasets").TestDataset
    rows=pd.read_csv("audit/step7_20261023_041838/protocol_lock/coordinate_diagnostic_manifest.csv")
    manifest=read_manifest("protocol/paired_episode_manifest.csv")
    for r in rows.to_dict("records"):
        ds=Dataset({"dataset":"SABS","data_dir":str(data_dir.parent),"eval_fold":int(r["fold"]),"supp_idx":2})
        ds.label=int(r["label_id"]);idx=next(i for i,p in enumerate(ds.image_dirs) if Path(p).name.split(".nii")[0]==r["patient_id"])
        original=ds[idx];support=ds.getSupport(label=int(r["label_id"]),all_slices=False,N=3)
        row=manifest[manifest.episode_id==r["episode_id"]].iloc[0].to_dict()
        case=next(x for x in episode_inputs(row,str(data_dir),config) if x["native_z"]==r["native_z"])
        np.testing.assert_array_equal(case["query"],original["image"][[r["compact_z"]]].float().numpy())
        np.testing.assert_array_equal(case["supp"],support["image"][[r["support_part"]]].float().numpy())
        np.testing.assert_array_equal(case["support_mask"],support["label"][[r["support_part"]]].float().numpy())
        np.testing.assert_array_equal(case["gt"],original["label"][r["compact_z"]].numpy()>0)

def test_legacy_and_restored_configurations_have_identical_pool():
    neg=np.array([[0.,255.],[123.125,234.5]],np.float32);pos=np.array([[12,23]],np.int64)
    variants,nr,pr=configurations(neg,pos)
    np.testing.assert_array_equal(variants["R1"][0],neg)
    np.testing.assert_array_equal(variants["R2"][0],nr)
    np.testing.assert_array_equal(variants["R2"][1],variants["R3"][1])
    assert len(variants["R3"][0])==0 and len(neg)==2

def test_exact_slice_roundtrip_and_resume_checksum(tmp_path):
    arrays={"R1_raw":np.array([[True,False],[False,True]]),"negative_source_xy":np.arange(20,dtype=np.float32).reshape(10,2)}
    record={"compact_z":0,"native_z":7,"episode_id":3,"generation_order_ids":list(range(10)),"seed":42}
    saved=save_slice(tmp_path,arrays,record)
    done=load_completed(tmp_path);assert done[0]["native_z"]==7
    with np.load(tmp_path/saved["file"]) as npz:
        np.testing.assert_array_equal(npz["negative_source_xy"],arrays["negative_source_xy"])
        assert json.loads(str(npz["record_json"]))==record
    with (tmp_path/saved["file"]).open("ab") as f:f.write(b"deliberate-corruption")
    with pytest.raises(AssertionError):load_completed(tmp_path)
