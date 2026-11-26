import json
from pathlib import Path

t = Path(__file__).resolve().parent
src = (t / 'worker.py').read_text()
compile(src, 't092_worker.py', 'exec')
code = (
    "from pathlib import Path\nimport subprocess, sys\n"
    f"Path('/kaggle/working/t092_worker.py').write_text({src!r})\n"
    "stages = [['prepare']] + [\n"
    "    [fold, cfg]\n"
    "    for fold in ['earlier', 'primary']\n"
    "    for cfg in ['xgb_d6', 'xgb_d5', 'xgb_d7', 'cat_d6', 'cat_d8']\n"
    "]\n"
    "for args in stages:\n"
    "    print('=== STAGE', args, flush=True)\n"
    "    subprocess.run([sys.executable, '-u', '/kaggle/working/t092_worker.py', *args], check=True)\n"
)
compile(code, 'run', 'exec')
nb = {
    'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': code.splitlines(True)}],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 4,
}
(t / 'notebook.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
(t / 'kernel-metadata.json').write_bytes((t.parent / '086-row-only-match-landmarks/kernel-metadata.json').read_bytes())
print('Built T092 notebook successfully')
