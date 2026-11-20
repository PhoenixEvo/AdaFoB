"""Lossless candidate logging only; never selects prompts."""
import json
from pathlib import Path
import numpy as np
from .coordinates import restore_xy


def candidate_record(episode_id, native_z, negatives, positives, source_hw, native_hw,
                     selected_ids, decoder_calls, variant):
    neg, pos = np.asarray(negatives), np.asarray(positives)
    if neg.shape != (len(neg), 2) or pos.shape != (len(pos), 2):
        raise ValueError("N,2 coordinates required")
    if not np.isfinite(neg).all() or not np.isfinite(pos).all():
        raise ValueError("nonfinite coordinates")
    ids = [int(i) for i in selected_ids]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(neg) for i in ids):
        raise ValueError("invalid immutable candidate IDs")
    if not isinstance(decoder_calls, int) or decoder_calls < 0:
        raise ValueError("decoder calls must be a nonnegative actual count")
    return {"episode_id": int(episode_id), "native_z": None if native_z is None else int(native_z), "variant": variant,
            "source_hw": list(source_hw), "native_hw": list(native_hw),
            "negative_dtype": str(neg.dtype), "positive_dtype": str(pos.dtype),
            "negatives_source_xy": neg.tolist(), "positives_source_xy": pos.tolist(),
            "negatives_native_restored_xy": restore_xy(neg, source_hw, native_hw).tolist(),
            "positives_native_restored_xy": restore_xy(pos, source_hw, native_hw).tolist(),
            "generation_order_ids": list(range(len(neg))), "selected_ids": ids,
            "actual_positive_count": len(pos), "actual_negative_count": len(ids),
            "emitted_negative_count": len(neg), "actual_total_count": len(pos) + len(ids),
            "decoder_evaluations": decoder_calls}


def write_records(path, records):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def source_arrays(record):
    return (np.asarray(record["negatives_source_xy"], dtype=record["negative_dtype"]).reshape(-1, 2),
            np.asarray(record["positives_source_xy"], dtype=record["positive_dtype"]).reshape(-1, 2))

class CandidateRunLog:
    """Logging only. Call sites provide immutable IDs; never infer or choose them."""
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.open("x",encoding="utf-8").close()
        self.current=None

    def begin_slice(self,episode_id,slice_index,negatives,positives,native_hw):
        if self.current is not None:
            raise RuntimeError("previous slice not flushed")
        self.current=candidate_record(episode_id,None,negatives,positives,(256,256),native_hw,[],0,"legacy_identity")
        self.current.update(slice_index=int(slice_index),slice_index_basis="GT-filtered active-slab index; native_z not inferred",
                            decoder_events=[],final_selected_ids_by_arm={})

    def decoder_event(self,selected_ids,arm,evaluated=True):
        ids=[int(v) for v in selected_ids]
        count=self.current["emitted_negative_count"]
        if len(set(ids))!=len(ids) or any(v<0 or v>=count for v in ids):
            raise ValueError("invalid selected IDs")
        event={"arm":str(arm),"selected_ids":ids,"actual_negative_count":len(ids),
               "actual_positive_count":self.current["actual_positive_count"],
               "actual_total_count":len(ids)+self.current["actual_positive_count"],
               "decoder_evaluations":int(evaluated),"outcome":"attempted" if evaluated else "empty_prompt_fallback"}
        self.current["decoder_events"].append(event)
        self.current["decoder_evaluations"]+=int(evaluated)
        self.current["final_selected_ids_by_arm"][str(arm)]=ids
        return event

    def final_choice(self,arm,selected_ids):
        self.current["final_selected_ids_by_arm"][str(arm)]=[int(v) for v in selected_ids]

    def finish_slice(self):
        if self.current is None:raise RuntimeError("no current slice")
        # Counts at the pool row refer to emitted full pool. Exact subset counts
        # belong to each decode event, including retries and GT-guided trials.
        self.current["selected_ids"]=list(range(self.current["emitted_negative_count"]))
        self.current["actual_negative_count"]=self.current["emitted_negative_count"]
        self.current["actual_total_count"]=self.current["emitted_negative_count"]+self.current["actual_positive_count"]
        with self.path.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(self.current,allow_nan=False,separators=(",",":"))+"\n")
        self.current=None
