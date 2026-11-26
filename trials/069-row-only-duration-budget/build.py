import json
from pathlib import Path
t=Path(__file__).resolve().parent;prev=t.parent/'063-row-only-embedding-depth';nb=json.loads((prev/'notebook.ipynb').read_text());code='from pathlib import Path\nimport subprocess,sys\n'
for path,name in [(t.parent/'062-row-only-gpu-replays/worker.py','base62.py'),(prev/'worker.py','rich63.py'),(t/'worker.py','t069.py')]:code+=f"Path('/kaggle/working/{name}').write_text({path.read_text()!r})\n"
code+="for args in [['prepare']]+[[f,str(e)] for f in ['earlier','primary'] for e in [0,1]]:\n    subprocess.run([sys.executable,'-u','/kaggle/working/t069.py',*args],check=True)\n";nb['cells'][0]['source']=code.splitlines(True);(t/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(t/'kernel-metadata.json').write_bytes((prev/'kernel-metadata.json').read_bytes())
