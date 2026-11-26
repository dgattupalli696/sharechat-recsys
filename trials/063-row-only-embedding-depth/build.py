import json
from pathlib import Path
t=Path(__file__).resolve().parent;prev=t.parent/'062-row-only-gpu-replays';code='from pathlib import Path\nimport subprocess, sys\n'
for src,name in [(prev/'worker.py','base62.py'),(t/'worker.py','trial63.py')]:code+=f"Path('/kaggle/working/{name}').write_text({src.read_text()!r})\n"
code+="for args in [['prepare']]+[[fold,str(depth),str(rich)] for fold in ['primary','earlier'] for depth,rich in [(12,0),(12,1),(6,0),(6,1)]]:\n    print(args,flush=True)\n    subprocess.run([sys.executable,'-u','/kaggle/working/trial63.py',*args],check=True)\n";nb=json.loads((prev/'notebook.ipynb').read_text());nb['cells'][0]['source']=code.splitlines(True);(t/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(t/'kernel-metadata.json').write_bytes((prev/'kernel-metadata.json').read_bytes())
