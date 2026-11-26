import json
from pathlib import Path

t = Path(__file__).resolve().parent
b92 = (t.parent / '092-super-enriched-trees/worker.py').read_text()
w93 = (t / 'worker.py').read_text()
compile(b92, 't092_worker.py', 'exec')
compile(w93, 't093_worker.py', 'exec')

code = (
    "from pathlib import Path\nimport subprocess, sys\n"
    f"Path('/kaggle/working/t092_worker.py').write_text({b92!r})\n"
    f"Path('/kaggle/working/t093_worker.py').write_text({w93!r})\n"
    "stages = [['prepare']] + [\n"
    "    [fold, arm]\n"
    "    for fold in ['earlier', 'primary']\n"
    "    for arm in ['xgb_d4_stat', 'xgb_d5_long', 'xgb_d5_rank']\n"
    "]\n"
    "for args in stages:\n"
    "    print('=== STAGE', args, flush=True)\n"
    "    subprocess.run([sys.executable, '-u', '/kaggle/working/t093_worker.py', *args], check=True)\n"
)
compile(code, 'run', 'exec')
nb = {
    'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': code.splitlines(True)}],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 4,
}
(t / 'notebook.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
(t / 'kernel-metadata.json').write_bytes((t.parent / '086-row-only-match-landmarks/kernel-metadata.json').read_bytes())
print('Built T093 notebook successfully')
