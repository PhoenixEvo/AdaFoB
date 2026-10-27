"""I/O-only prediction wrapper: persist and verify BEFORE caller evaluates GT."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def array_hashes(arrays):
    return {k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
            for k,v in arrays.items()}

def persistence_wrapper(predict,output,clock=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);index=0
    def wrapped(**legitimate):
        nonlocal index
        arrays,record=predict(**legitimate) # Frozen API rejects GT/oracle keys.
        case='case'+str(index)
        if index>=8:raise RuntimeError('Persistence case cap')
        def persist():
            partial=root/(case+'.partial.npz');final=root/(case+'.npz')
            with partial.open('xb') as f:
                np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            os.rename(partial,final)
            expected=array_hashes(arrays)
            with np.load(final,allow_pickle=False) as z:
                if set(z.files)!=set(expected):raise RuntimeError('Persistence keys')
                actual=array_hashes({k:z[k] for k in z.files})
            if actual!=expected:raise RuntimeError('Persisted arrays changed')
            # This seal exists before prediction returns to any metric caller.
            seal={'status':'label_free_predictions_persisted_and_reloaded','case':case,
                  'sealed_unix':time.time(),'npz_sha256':sha(final),'array_hashes':actual,
                  'prediction_record':record,'retrospective_evaluation':'NOT STARTED',
                  'no_query_gt_received':True}
            with (root/(case+'.seal.json')).open('x',encoding='utf-8') as f:
                json.dump(seal,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        if clock is None:persist()
        else:clock.call('label_free_persistence_and_integrity',persist)
        index+=1
        return arrays,record # No masks, choices, scores or metrics changed.
    return wrapped