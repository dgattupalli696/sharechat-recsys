"""Build the three missing full fits from the frozen T063 code."""
import json
from pathlib import Path
t=Path(__file__).resolve().parent;prev=t.parent/'063-row-only-embedding-depth';nb=json.loads((prev/'notebook.ipynb').read_text());src=(prev/'worker.py').read_text();code='from pathlib import Path\nimport subprocess, sys\n';base=(t.parent/'062-row-only-gpu-replays/worker.py').read_text()
def replace(old,new):
    global src
    assert src.count(old)==1;src=src.replace(old,new)
replace("/working/t063","/working/t066");replace("/temp/t063","/temp/t066")
replace('def prepare(data):','def prepare(data, split):');replace("raw=data/'train.parquet'","raw=data/f'{split}.parquet'");replace('assert n==1537638','assert n==(1537638 if split==\'train\' else 1037090)');replace("CACHE/'x.npy',mode='w+'","CACHE/f'{split}.npy',mode='w+'");replace("m.flush();f=pl.read_parquet", "m.flush()\n    if split=='test':return\n    f=pl.read_parquet")
i=src.index(";cut=base.CUTOFF");j=src.index('; width=',i);src=src[:i]+";tr=np.arange(len(y));assert len(tr)==1537638"+src[j:];replace("CACHE/'x.npy',mmap_mode", "CACHE/'train.npy',mmap_mode");replace('def pred(j):',"x=np.load(CACHE/'test.npy',mmap_mode='r');va=np.arange(len(x));assert len(va)==1037090\n    def pred(j):");i=src.index(';np.savez_compressed(OUT/');j=src.index('\n',i);src=src[:i]+";np.save(OUT/f'{key}.npy',p);return {'fit':len(tr),'test':len(va),'width':width}"+src[j:];replace("if s==['prepare']:prepare(next(p.parent for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists()));report={}","if s[0]=='prepare':prepare(next(p.parent for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists()),s[1]);report={}");compile(src,'refit.py','exec');(t/'worker.py').write_text(src)
for name,text in [('base62.py',base),('refit.py',src)]:code+=f"Path('/kaggle/working/{name}').write_text({text!r})\n"
code+="for args in [['prepare','train'],['prepare','test'],['full','12','1'],['full','6','0'],['full','6','1']]:\n    subprocess.run([sys.executable,'-u','/kaggle/working/refit.py',*args],check=True)\n";nb['cells'][0]['source']=code.splitlines(True);(t/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(t/'kernel-metadata.json').write_bytes((prev/'kernel-metadata.json').read_bytes())
