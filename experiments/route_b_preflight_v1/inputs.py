"""Standard-reader manifest verification and approved-role resolution, no inference."""
from pathlib import Path
import csv,hashlib,gzip,struct,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def roles(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.reader(f))
    if len(records)!=9 or any(len(r)!=10 for r in records):raise ValueError('Entitlement shape')
    rows=[dict(zip(records[0],r)) for r in records[1:]]
    desired={'construction_amended_Route_B':'image_8','check_amended_Route_B':'image_20','support_amended_Route_B':'image_14'}
    for role,patient in desired.items():
        matched=[r for r in rows if r['role']==role]
        if len(matched)!=1 or matched[0]['patient']!=patient or not matched[0]['status'].startswith('APPROVED'):raise ValueError('Amended role unresolved')
    return desired

def manifest_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=8:raise ValueError('Eight locked rows required')
    expected={(p,c) for p in ('image_8','image_20') for c in (4,7,8,11)}
    if {(r['patient'],int(r['class_id'])) for r in rows}!=expected:raise ValueError('Unexpected tasks')
    for r in rows:
        if r['role']!=('construction' if r['patient']=='image_8' else 'check') or r['support_patient']!='image_14':raise ValueError('Role mismatch')
    return rows

def read_volume(path):
    import SimpleITK as sitk
    im=sitk.ReadImage(str(path));a=sitk.GetArrayFromImage(im)
    # Independently compare decoded values against the NIfTI1 payload (no rescaling here).
    with gzip.open(path,'rb') as f:raw=f.read()
    if struct.unpack_from('<i',raw,0)[0]!=348:raise ValueError('Little-endian NIfTI1 required')
    dim=struct.unpack_from('<8h',raw,40);typ=struct.unpack_from('<h',raw,70)[0]
    off=int(struct.unpack_from('<f',raw,108)[0]);slope,inter=struct.unpack_from('<2f',raw,112)
    if slope not in (0.,1.) or inter!=0:raise ValueError('Scaled NIfTI not locked')
    dtype={2:'u1',4:'<i2',8:'<i4',16:'<f4',64:'<f8',256:'i1',512:'<u2'}[typ]
    b=np.frombuffer(raw,dtype=dtype,count=int(np.prod(dim[1:4])),offset=off).reshape(tuple(reversed(dim[1:4])))
    np.testing.assert_array_equal(a,b)
    if dim[0]!=3 or tuple(im.GetSize())!=tuple(dim[1:4]):raise ValueError('Dimensions')
    if raw[123]&7!=2:raise ValueError('Physical units not mm')
    sf=struct.unpack_from('<h',raw,254)[0]
    if sf<=0:raise ValueError('Missing sform')
    ras=np.eye(4);ras[:3]=np.array(struct.unpack_from('<12f',raw,280)).reshape(3,4)
    lps=np.eye(4);lps[:3,:3]=np.array(im.GetDirection()).reshape(3,3)@np.diag(im.GetSpacing());lps[:3,3]=im.GetOrigin()
    from_sitk=np.diag([-1.,-1.,1.,1.])@lps
    np.testing.assert_allclose(from_sitk,ras,atol=1e-6,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(ras[:3,:3],axis=0),im.GetSpacing(),atol=1e-6,rtol=0)
    return {'array':a,'dtype':dtype,'itk':im,'ras':ras,'file_sha256':sha(path),
            'decoded_array_sha256':hashlib.sha256(np.ascontiguousarray(a,dtype=dtype).tobytes()).hexdigest()}

def verify_manifest(manifest,data_dir,entitlements):
    roles(entitlements);rows=manifest_rows(manifest);volumes={};results=[]
    for patient in ('image_8','image_20','image_14'):
        image=read_volume(Path(data_dir)/(patient+'.nii.gz'))
        label=read_volume(Path(data_dir)/(patient.replace('image_','label_')+'.nii.gz'))
        np.testing.assert_allclose(image['ras'],label['ras'],atol=1e-6,rtol=0)
        if image['array'].shape!=label['array'].shape:raise ValueError('Image/label alignment')
        volumes[patient]=(image,label)
    for r in rows:
        c=int(r['class_id']);case={'patient':r['patient'],'class_id':c}
        for prefix,patient,zkey in [('query',r['patient'],'slice_index'),('support',r['support_patient'],'support_slice_index')]:
            image,label=volumes[patient];z=int(r[zkey]);a=label['array'];extent=np.flatnonzero((a==c).any(axis=(1,2)))
            if not len(extent) or int(extent[(len(extent)-1)//2])!=z:raise ValueError('Slice median mismatch')
            stem='' if prefix=='query' else 'support_'
            dims=tuple(int(x) for x in r[stem+'native_xyz'].split(';'))
            if dims!=tuple(image['itk'].GetSize()):raise ValueError('Manifest dimensions')
            np.testing.assert_allclose(image['itk'].GetSpacing(),list(map(float,r[stem+'spacing_xyz'].split(';'))),atol=1e-6,rtol=0)
            # Manifest geometry is decimal-rounded to six places; compare within 1e-6 mm.
            np.testing.assert_allclose(image['ras'][:3].ravel(),list(map(float,r[stem+'affine_sform'].split(';'))),atol=1e-6,rtol=0)
            mask=(a[z]==c).astype(np.uint8)
            img_slice=np.ascontiguousarray(image['array'][z],dtype=image['dtype'])
            checks={prefix+'_image_sha256':image['file_sha256'],prefix+'_label_sha256':label['file_sha256'],
                    prefix+'_image_slice_sha256':hashlib.sha256(img_slice.tobytes(order='C')).hexdigest(),
                    prefix+'_base_mask_sha256':hashlib.sha256(mask.tobytes(order='C')).hexdigest()}
            for key,value in checks.items():
                if value.lower()!=r[key].lower():raise ValueError('Locked hash mismatch: '+key)
            case[prefix]={'z':z,'checks':checks,'array_order':'zyx volume; yx slice; x fastest',
                          'slice_serialization':'native stored scalar type; little endian; C-contiguous; no normalization',
                          'mask_serialization':'uint8 0/1; C-contiguous yx',
                          'decoded_image_array_sha256':image['decoded_array_sha256'],
                          'decoded_label_array_sha256':label['decoded_array_sha256']}
        results.append(case)
    return results,rows,volumes

def prepared_cases(rows,volumes):
    """Assembly assists with locked slice choice; evaluation labels stay separate."""
    from experiments.utility_feasibility_v1.slice_api import Geometry,FrozenConfig
    normalized={}
    for p,(image,label) in volumes.items():
        v=image['array'];normalized[p]=((v-v.mean())/v.std()).astype(np.float32)
    for i,r in enumerate(rows):
        p=r['patient'];s=r['support_patient'];c=int(r['class_id']);z=int(r['slice_index']);sz=int(r['support_slice_index'])
        im=volumes[p][0]['itk']
        legitimate=dict(query_image=np.repeat(normalized[p][z][None,None],3,axis=1),
                        support_image=np.repeat(normalized[s][sz][None,None],3,axis=1),
                        support_mask=(volumes[s][1]['array'][sz]==c)[None].astype(np.float32),
                        geometry=Geometry(native_hw=normalized[p].shape[-2:],spacing_xy=im.GetSpacing()[:2],
                                          origin_xyz=im.TransformIndexToPhysicalPoint((0,0,z)),direction_xyz=im.GetDirection()),configuration=FrozenConfig())
        yield 'case'+str(i),legitimate,volumes[p][1]['array'][z]==c

def execute_prepared(rows,volumes,backend,writer):
    """Inference entry for later authorization only. NOT called in CPU preflight."""
    from experiments.route_b_preflight_v1.runner import pilot_predict,evaluate_saved
    for case_id,legitimate,query_gt in prepared_cases(rows,volumes):
        arrays,record=pilot_predict(**legitimate,backend=backend)
        record['evaluation']=evaluate_saved(arrays,query_gt)
        writer.save(case_id,arrays,record)
    writer.complete()