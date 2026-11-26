from pathlib import Path
import json
t=Path(__file__).resolve().parent;prev=t.parent/'073-compact-tree-resolution';s=(prev/'worker.py').read_text().replace("/t073'","/t079'");s=s.replace('from t069 import duration','from t069 import duration\nimport encoder79 as enc');s=s.replace("width=BASE; bins=int(arm.split('-')[1]); leaf=arm.startswith('leaf-')", """width=BASE;bins=127;leaf=False
    fields=enc.FIELDS[:11] if arm=='viewer' else enc.FIELDS[11:] if arm=='content' else enc.FIELDS
    raw=next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists());extras=[]
    for k in fields:
        col=f'feature{k}';s=pl.read_parquet(raw,columns=[col])[col];tab=enc.learn(s.gather(tr)); e=enc.encode(s,tab);np.testing.assert_array_equal(e[va[:8]],enc.encode(s.gather(va[:8]),tab));extras.append(e.astype('uint8'))
    extra=np.column_stack(extras);del extras
    def matrix(j):return np.column_stack([x[j,:width],extra[j]])
    """);s=s.replace('cp.asarray(x[j,:width])','cp.asarray(matrix(j))');compile(s,'worker','exec');(t/'worker.py').write_text(s);nb=json.loads((prev/'notebook.ipynb').read_text());code=''.join(nb['cells'][0]['source']).split("Path('/kaggle/working/t073.py').write_text(",1)[0];
for name,body in [('encoder79.py',(t/'encoder.py').read_text()),('t079.py',s)]:code+=f"Path('/kaggle/working/{name}').write_text({body!r})\n"
code+="for args in [['prepare']]+[[fold,arm] for fold in ['earlier','primary'] for arm in ['viewer','content','both']]:\n    subprocess.run([sys.executable,'-u','/kaggle/working/t079.py',*args],check=True)\n";compile(code,'nb','exec');nb['cells'][0]['source']=code.splitlines(True);(t/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(t/'kernel-metadata.json').write_bytes((prev/'kernel-metadata.json').read_bytes());(t/'score.py').write_text((prev/'score.py').read_text().replace("['depth-255','leaf-127','leaf-255']","['viewer','content','both']"))
