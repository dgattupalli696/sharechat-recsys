"""Build Kaggle GPU v60 artifact-export notebook for all 15 T062/T066/T087/T096 S025 models."""
import json
from pathlib import Path

t = Path(__file__).resolve().parent
t014_nb = json.loads((t.parent / '014-full-gpu-combinations-architectures/notebook.ipynb').read_text())
wheels_cell = dict(t014_nb['cells'][2])
wheels_cell['source'] = ['import sys\n', 'from pathlib import Path\n'] + list(wheels_cell['source'])

files = [
    ('base62.py', t.parent / '062-row-only-gpu-replays/worker.py'),
    ('rich63.py', t.parent / '063-row-only-embedding-depth/worker.py'),
    ('t069.py', t.parent / '069-row-only-duration-budget/worker.py'),
    ('t070.py', t.parent / '070-row-only-duration-neural/worker.py'),
    ('quantiles81.py', t.parent / '081-row-only-quantile-neural/quantiles.py'),
    ('rot83.py', t.parent / '083-row-only-paired-rotations/rotations.py'),
    ('land86.py', t.parent / '086-row-only-match-landmarks/landmarks.py'),
    ('t092_worker.py', t.parent / '092-super-enriched-trees/worker.py'),
    ('t094_worker.py', t.parent / '094-neural-diversity-and-rank/worker.py'),
    ('t095_worker.py', t.parent / '095-quantile-plr-neural/worker.py'),
    ('t098_worker.py', t / 'worker.py'),
]
run_code = "from pathlib import Path\nimport subprocess, sys, os\n"
for name, path in files:
    src = path.read_text()
    compile(src, name, 'exec')
    run_code += f"Path('/kaggle/working/{name}').write_text({src!r})\n"

run_code += (
    "env = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3')\n"
    "stages = [\n"
    "    ['prepare'],\n"
    "    ['t062_d12_r0'], ['t066_d12_r1'], ['t066_d6_r0'], ['t066_d6_r1'],\n"
    "    ['t087_duration'], ['t087_compact'], ['t087_paired'], ['t087_learned'], ['t087_random'],\n"
    "    ['t087_gated'], ['t087_quantile'],\n"
    "    ['t096_xgb_d6'], ['t096_dcnv2_r64', '42'], ['t096_q_tabm', '42'], ['t096_q_tabm', '43'],\n"
    "]\n"
    "for args in stages:\n"
    "    print('=== EXPORT', args, flush=True)\n"
    "    subprocess.run([sys.executable, '-u', '/kaggle/working/t098_worker.py', *args], env=env, check=True)\n"
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
print('Built T098 artifact-export notebook successfully')
