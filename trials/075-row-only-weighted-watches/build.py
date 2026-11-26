"""Package two predeclared weighted logistic fits using the frozen T072 compact design."""
import json
from pathlib import Path
t=Path(__file__).resolve().parent;prev=t.parent/'073-compact-tree-resolution';src=(prev/'worker.py').read_text().replace("/t073'","/t075'")
def edit(old,new):
    global src
    assert src.count(old)==1;src=src.replace(old,new)
edit("bins=int(arm.split('-')[1]); leaf=arm.startswith('leaf-')","bins=127; leaf=False; ratio={'pos2':2.,'pos5':5.}[arm]; norm=1+(ratio-1)*float(y[tr].mean())")
edit('label=cp.asarray(y[j])','label=cp.asarray(y[j]),weight=cp.asarray(np.where(y[j]>0,ratio,1.).astype(np.float32)/np.float32(norm))')
edit('def predict(j):return cp.asnumpy(model.inplace_predict(cp.asarray(x[j,:width])))','def predict(j):\n            q=cp.asnumpy(model.inplace_predict(cp.asarray(x[j,:width]))).astype(np.float64);return q/(ratio*(1-q)+q)')
compile(src,'t075','exec');(t/'worker.py').write_text(src);nb=json.loads((prev/'notebook.ipynb').read_text());code=''.join(nb['cells'][0]['source']).split("Path('/kaggle/working/t073.py').write_text(",1)[0];code+=f"Path('/kaggle/working/t075.py').write_text({src!r})\nfor args in [['prepare']]+[[fold,arm] for fold in ['earlier','primary'] for arm in ['pos2','pos5']]:\n    subprocess.run([sys.executable,'-u','/kaggle/working/t075.py',*args],check=True)\n";compile(code,'notebook','exec');nb['cells'][0]['source']=code.splitlines(True);(t/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(t/'kernel-metadata.json').write_bytes((prev/'kernel-metadata.json').read_bytes());score=(prev/'score.py').read_text().replace("['depth-255','leaf-127','leaf-255']","['pos2','pos5']");(t/'score.py').write_text(score)
