import json
from pathlib import Path

t = Path(__file__).resolve().parent
t014_nb = json.loads((t.parent / '014-full-gpu-combinations-architectures/notebook.ipynb').read_text())
wheels_cell = dict(t014_nb['cells'][2])
wheels_cell['source'] = ['import sys\n', 'from pathlib import Path\n'] + list(wheels_cell['source'])

b92 = (t.parent / '092-super-enriched-trees/worker.py').read_text()
w94 = (t / 'worker.py').read_text()
compile(b92, 't092_worker.py', 'exec')
compile(w94, 't094_worker.py', 'exec')

run_code = (
    "from pathlib import Path\nimport subprocess, sys\n"
    f"Path('/kaggle/working/t092_worker.py').write_text({b92!r})\n"
    f"Path('/kaggle/working/t094_worker.py').write_text({w94!r})\n"
    "stages = [['prepare']] + [\n"
    "    [fold, arm]\n"
    "    for fold in ['earlier', 'primary']\n"
    "    for arm in ['dcnv2_r64', 'dcnv2_moe4', 'finalmlp', 'rank_d5', 'rank_d6']\n"
    "]\n"
    "for args in stages:\n"
    "    print('=== STAGE', args, flush=True)\n"
    "    subprocess.run([sys.executable, '-u', '/kaggle/working/t094_worker.py', *args], check=True)\n"
)
compile(run_code, 'run', 'exec')
nb = {
    'cells': [
        wheels_cell,
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': run_code.splitlines(True)},
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 4,
}
(t / 'notebook.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
(t / 'kernel-metadata.json').write_bytes((t.parent / '086-row-only-match-landmarks/kernel-metadata.json').read_bytes())
print('Built T094 notebook successfully')
