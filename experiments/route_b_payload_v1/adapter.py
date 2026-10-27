"""Owner-approved payload adapter. No frozen file is edited.
Identity is complete plain NIfTI header+voxel bytes, not gzip container bytes.
"""
from pathlib import Path
import builtins,copy,json
from unittest.mock import patch
from experiments.route_b_preflight_v1 import inputs as frozen

_READ=frozen.read_volume
_ROWS=frozen.manifest_rows
_STEMS={f'{kind}_{i}' for i in (8,14,20) for kind in ('image','label')}

def load_identities(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    rows=obj['files']
    if len(rows)!=6 or {r['stem'] for r in rows}!=_STEMS:raise ValueError('Exactly six approved payload identities required')
    result={r['stem']:r for r in rows}
    for stem,r in result.items():
        if r['payload_name']!=stem+'.nii' or r['original_gzip_name']!=stem+'.nii.gz' or not r['original_gzip_matches_locked']:raise ValueError('Identity provenance')
        for key in ('payload_sha256','original_gzip_sha256'):
            if len(r[key])!=64 or any(c not in '0123456789abcdef' for c in r[key]):raise ValueError('Invalid SHA256')
    return result

def verify_payload_manifest(manifest,data_dir,entitlements,identity_file):
    """Reuse every frozen numeric/slice/geometry check; replace file identity only."""
    ids=load_identities(identity_file);data_dir=Path(data_dir)
    original_rows=_ROWS(manifest)
    amended=copy.deepcopy(original_rows)
    for row in amended:
        for prefix,patient in [('query',row['patient']),('support',row['support_patient'])]:
            for kind in ('image','label'):
                stem=patient if kind=='image' else patient.replace('image_','label_')
                key=prefix+'_'+kind+'_sha256'
                if row[key].lower()!=ids[stem]['original_gzip_sha256']:raise ValueError('Original container provenance changed')
                row[key]=ids[stem]['payload_sha256']
    evidence=[]
    for stem in sorted(ids):
        path=data_dir/ids[stem]['payload_name']
        actual=frozen.sha(path)
        if actual!=ids[stem]['payload_sha256'] or path.stat().st_size!=ids[stem]['payload_bytes']:raise ValueError('PAYLOAD MISMATCH: '+stem)
        evidence.append({'stem':stem,'sha256':actual,'bytes':path.stat().st_size,'identity_basis':'complete decompressed NIfTI payload',
                         'gzip_container_check_on_plain_file':'NOT APPLICABLE; not claimed passed'})
    allowed={(data_dir/(stem+'.nii')).resolve() for stem in ids}
    def plain_open(path,mode='rb',*a,**kw):
        if Path(path).resolve() not in allowed or mode!='rb':raise ValueError('Adapter I/O outside six approved plain files')
        return builtins.open(path,mode,*a,**kw)
    def adapted_read(requested):
        requested=Path(requested)
        if requested.name not in {s+'.nii.gz' for s in ids}:raise ValueError('Unsupported requested volume')
        path=requested.with_suffix('')
        # The frozen reader still executes its exact SimpleITK, header, scaling,
        # sform, RAS/LPS, dimension and decoded-array checks. Only its gzip.open
        # binding is replaced for these explicitly allowlisted plain paths.
        with patch.object(frozen.gzip,'open',plain_open):return _READ(path)
    with patch.object(frozen,'read_volume',adapted_read),patch.object(frozen,'manifest_rows',lambda unused:copy.deepcopy(amended)):
        cases,_,volumes=frozen.verify_manifest(manifest,data_dir,entitlements)
    return {'identity_amendment':'gzip container SHA256 -> full decompressed NIfTI payload SHA256, owner approved',
            'payloads':evidence,'cases':cases},original_rows,volumes