"""Build Kaggle GPU v59 full-data refit notebook for S025 (Step 6 of T095: 7/11 S024 + 2/11 xgb_d6 + 1/11 q_tabm_2s_ep8 + 1/11 dcnv2_r64)."""
import json
from pathlib import Path

t = Path(__file__).resolve().parent
t014_nb = json.loads((t.parent / '014-full-gpu-combinations-architectures/notebook.ipynb').read_text())
wheels_cell = dict(t014_nb['cells'][2])
wheels_cell['source'] = ['import sys\n', 'from pathlib import Path\n'] + list(wheels_cell['source'])

files = [
    ('t092_worker.py', t.parent / '092-super-enriched-trees/worker.py'),
    ('t094_worker.py', t.parent / '094-neural-diversity-and-rank/worker.py'),
    ('quantiles81.py', t.parent / '081-row-only-quantile-neural/quantiles.py'),
    ('t095_worker.py', t.parent / '095-quantile-plr-neural/worker.py'),
    ('t096_worker.py', t / 'worker.py'),
]
run_code = "from pathlib import Path\nimport subprocess, sys\n"
for name, path in files:
    src = path.read_text()
    compile(src, name, 'exec')
    run_code += f"Path('/kaggle/working/{name}').write_text({src!r})\n"

run_code += (
    "stages = [\n"
    "    ['prepare'],\n"
    "    ['xgb_d6'],\n"
    "    ['dcnv2_r64', '42'],\n"
    "    ['q_tabm', '42'],\n"
    "    ['q_tabm', '43'],\n"
    "]\n"
    "for args in stages:\n"
    "    print('=== STAGE', args, flush=True)\n"
    "    subprocess.run([sys.executable, '-u', '/kaggle/working/t096_worker.py', *args], check=True)\n"
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
print('Built T096 full-refit notebook successfully')
