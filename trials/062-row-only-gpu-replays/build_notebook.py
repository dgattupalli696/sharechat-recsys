import json
from pathlib import Path

t=Path(__file__).resolve().parent
code="from pathlib import Path\nimport subprocess, sys\nworker=Path('/kaggle/working/t062-worker.py')\nworker.write_text("+repr((t/'worker.py').read_text())+")\nfor stage in ['prepare','depth12','depth12-full','leaves511','catboost-ids']:\n    print(stage,flush=True)\n    subprocess.run([sys.executable,'-u',str(worker),stage],check=True)\n"
(t/'notebook.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':4},indent=1)+'\n')
