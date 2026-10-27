import ast,hashlib,io,json
from pathlib import Path
import numpy as np,pandas as pd
from analysis.controls_v1.core import fps,proximity,random_ids,references,dice,BASE,ZERO,GREEDY_REF
from analysis.controls_v1.statistics import ELIGIBLE,gap_analysis
from analysis.rebaseline_v1.common import serialized_slice

def test_permutation_invariant_standard_controls():
    p=np.array([[0,0],[1,1],[1,1],[0,4],[4,0],[4,4],[2,2],[2,0],[0,2],[3,3]],float);ids=np.arange(10)
    mask=np.zeros((6,6),bool);mask[1:5,1:5]=1
    for k in [2,4,6]:
        for perm in [np.arange(10)[::-1],np.random.default_rng(42).permutation(10)]:
            assert fps(p,ids,k)==fps(p[perm],ids[perm],k)
            for far in [False,True]:assert proximity(p,ids,k,mask,far)==proximity(p[perm],ids[perm],k,mask,far)
            assert random_ids(ids,k,43,7,12)==random_ids(ids[perm],k,43,7,12)

def test_empty_predicted_boundary_and_nested_random():
    p=np.zeros((10,2));ids=np.arange(10)
    assert proximity(p,ids,4,np.zeros((7,7),bool))==[0,1,2,3]
    assert proximity(p,ids,4,np.zeros((7,7),bool),True)==[0,1,2,3]
    assert random_ids(ids,2,42,0,3)==random_ids(ids,6,42,0,3)[:2]

def test_reference_actual_legacy_ast_parity():
    source=ast.parse(Path('experiments/run_step5_diagnostics.py').read_text())
    f=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='compute_slice_dice')
    prefix=next(n for n in ast.walk(source) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='k_cand')
    greedy=next(n for n in ast.walk(source) if isinstance(n,ast.While) and isinstance(n.test,ast.Name) and n.test.id=='remaining_indices')
    for seed in range(20):
        gt=np.random.default_rng(seed).random((12,12))>.6
        def decode(ids):
            h=int.from_bytes(hashlib.sha256(str((seed,tuple(ids))).encode()).digest()[:8],'little')
            return np.random.default_rng(h).random(gt.shape)>.55
        context={'np':np,'gt_slice':gt,'m_k0':decode([]),'neg_arr':np.arange(10)[:,None],'pos_arr':[],
                 'H_q':12,'W_q':12,'predictor_b':None,'candidate_log':None}
        context['predict_singlemask']=lambda predictor,pos,neg,shape,**kwargs:decode(np.asarray(neg).ravel().tolist())
        exec(compile(ast.Module(body=[f],type_ignores=[]),'legacy_dice','exec'),context)
        context.update(best_dice=context['compute_slice_dice'](context['m_k0'],gt),best_mask_prefix=context['m_k0'],best_prefix_ids=[])
        exec(compile(ast.Module(body=[prefix],type_ignores=[]),'legacy_prefix','exec'),context)
        context.update(current_selected=[],current_best_dice=dice(context['m_k0'],gt),current_best_mask=context['m_k0'],remaining_indices=list(range(10)))
        exec(compile(ast.Module(body=[greedy],type_ignores=[]),'legacy_greedy','exec'),context)
        a,b,t=references(decode,gt)
        assert a==context['best_prefix_ids'] and b==context['current_selected']
        assert sum(len(s['trials']) for s in t['greedy'])<=55

def test_reference_ties_retain_zero_and_trajectory_roundtrip():
    mask=np.eye(4,dtype=bool)
    p,g,t=references(lambda ids:mask,mask)
    assert p==g==[] and len(t['prefix'])==11 and len(t['greedy'][0]['trials'])==10
    data=serialized_slice({'raw':mask},{'trajectories':t,'selected_ids':g})
    with np.load(io.BytesIO(data),allow_pickle=False) as saved:
        assert np.array_equal(saved['raw'],mask)
        assert json.loads(str(saved['record_json']))['trajectories']==t

def fixture(closure):
    rows=[]
    for patient in range(3):
        for organ in ['LIVER','LK','RK','SPLEEN']:
            for arm in ELIGIBLE+[GREEDY_REF]:
                score=.8 if arm==BASE else .82 if arm==GREEDY_REF else .8+.02*closure
                rows.append(dict(episode_id=patient*4+['LIVER','LK','RK','SPLEEN'].index(organ),fold=0,patient_id=str(patient),organ=organ,arm=arm,post='cc',dice=score,hd95=10 if arm==GREEDY_REF else 11,recall=.9))
    return pd.DataFrame(rows)
def test_locked_gap_decisions():
    assert gap_analysis(fixture(.6),100,42)['branch']=='STOP-SELECTOR'
    assert gap_analysis(fixture(.1),100,42)['branch']=='CONTINUE'
    assert gap_analysis(fixture(.4),100,42)['branch']=='REVIEW'
def test_zero_reference_gap_not_silently_divided():
    df=fixture(.4);df.loc[df.arm==GREEDY_REF,'dice']=.8
    r=gap_analysis(df,100,42)
    assert r['branch']=='REVIEW' and r['undefined_replicates']==100 and r['best_reselected_ci95'] is None
def test_worker_cpu_harness_and_resume(tmp_path,monkeypatch):
    """Exercise real worker persistence using CPU mocks, never a GPU test."""
    import sys,time,torch
    import analysis.controls_v1.inference as worker
    lock=tmp_path/'lock';lock.mkdir()
    (lock/'step8_fingerprints.json').write_text(json.dumps({'0:4':{'masks':'unused'},'0:5':{'masks':'unused'}}))
    pd.DataFrame([dict(episode_id=0,worker=0)]).to_csv(lock/'worker_assignment.csv',index=False)
    monkeypatch.setattr(worker,'read_config',lambda p:{'seed':42})
    monkeypatch.setattr(worker,'read_manifest',lambda p:pd.DataFrame([dict(episode_id=0,fold=0)]))
    monkeypatch.setattr(worker,'seed_all',lambda s:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda n:'Tesla T4 CPU MOCK')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(worker,'tracked_sam',lambda p:(None,{}))
    monkeypatch.setattr(worker,'load_models',lambda *args:(object(),{'checkpoint_sha256':'cpu_mock'}))
    cases=[dict(episode_id=0,compact_z=z,native_z=z+4,active_slices=2) for z in range(2)]
    monkeypatch.setattr(worker,'episode_inputs',lambda *args:iter(cases))
    calls=[]
    def fake_run(model,predictor,case,config,expected):
        assert isinstance(expected,dict)
        calls.append(case['compact_z'])
        return {'gt':np.zeros((2,2),bool)},{**case,'failures':[]}
    monkeypatch.setattr(worker,'run_slice',fake_run)
    out=tmp_path/'output'
    argv=['worker','--lock-dir',str(lock),'--data-dir','unused','--checkpoint-root','unused','--sam-checkpoint','unused','--output',str(out),'--worker','0','--deadline-epoch',str(time.time()+300)]
    monkeypatch.setattr(sys,'argv',argv);worker.main();worker.main()
    assert calls==[0,1]
    assert len(list(out.glob('episode_*/slice_*.commit.json')))==2
    assert len(list(out.glob('worker_complete_attempt*.json')))==2