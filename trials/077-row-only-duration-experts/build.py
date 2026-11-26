"""Freeze T072 compact training with matched duration/random partition experts."""
import json
from pathlib import Path
t=Path(__file__).resolve().parent;prev=t.parent/'073-compact-tree-resolution';src=(prev/'worker.py').read_text().replace("/t073'","/t077'");src=src.replace('assert off==len(x);x.flush()',"assert off==len(x);np.save(CACHE/'duration.npy',x[:,names.index('feature54')]);x.flush()");old="width=BASE; bins=int(arm.split('-')[1]); leaf=arm.startswith('leaf-')";assert old in src;src=src.replace(old,"""width=BASE; bins=127;leaf=False
    if arm!='pooled':
        kind,branch=arm.split('-');d=np.load(CACHE/'duration.npy');short=np.isfinite(d)&(d<=60);size=int(short[tr].sum());n=len(tr)
        if kind=='duration':tr=tr[short[tr] if branch=='short' else ~short[tr]]
        else:
            assert kind=='random';rnd=np.random.default_rng(42).permutation(tr);tr=np.sort(rnd[:size] if branch=='short' else rnd[size:])
        assert len(tr)==(size if branch=='short' else n-size)
    """);src=src.replace("'features':width}","'features':width,'train_rows':len(tr)}");compile(src,'t077','exec');(t/'worker.py').write_text(src);nb=json.loads((prev/'notebook.ipynb').read_text());code=''.join(nb['cells'][0]['source']).split("Path('/kaggle/working/t073.py').write_text(",1)[0]+f"Path('/kaggle/working/t077.py').write_text({src!r})\nfor args in [['prepare']]+[[fold,arm] for fold in ['earlier','primary'] for arm in ['duration-short','duration-long','random-short','random-long']]:\n    subprocess.run([sys.executable,'-u','/kaggle/working/t077.py',*args],check=True)\n";compile(code,'nb','exec');nb['cells'][0]['source']=code.splitlines(True);(t/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(t/'kernel-metadata.json').write_bytes((prev/'kernel-metadata.json').read_bytes())
