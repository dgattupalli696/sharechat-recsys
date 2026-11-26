"""Build Kaggle GPU notebook for T090 with T014's exact offline TabM/PLR wheels."""
import json
from pathlib import Path

t = Path(__file__).resolve().parent
t014_nb = json.loads((t.parent/'014-full-gpu-combinations-architectures/notebook.ipynb').read_text())
wheels_cell = dict(t014_nb['cells'][2])
wheels_cell['source'] = ['import sys\n', 'from pathlib import Path\n'] + list(wheels_cell['source'])
worker_src = (t/'worker.py').read_text()
compile(worker_src, 't090_worker.py', 'exec')

run_code = (
    "from pathlib import Path\nimport subprocess, sys\n"
    f"Path('/kaggle/working/t090_worker.py').write_text({worker_src!r})\n"
    "stages = [['prepare']] + [\n"
    "    [fold, arch, str(w), str(seed)]\n"
    "    for fold in ['earlier', 'primary']\n"
    "    for arch, w, seed in [\n"
    "        ('tabm', 128, 42), ('tabm', 128, 43),\n"
    "        ('tabm', 256, 42), ('tabm', 256, 43),\n"
    "        ('dcnv2', 128, 42),\n"
    "    ]\n"
    "]\n"
    "for args in stages:\n"
    "    print('=== STAGE', args, flush=True)\n"
    "    subprocess.run([sys.executable, '-u', '/kaggle/working/t090_worker.py', *args], check=True)\n"
)
compile(run_code, 'run_code', 'exec')
nb = {
    'cells': [
        wheels_cell,
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': run_code.splitlines(True)},
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 4,
}
(t/'notebook.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
(t/'kernel-metadata.json').write_bytes((t.parent/'086-row-only-match-landmarks/kernel-metadata.json').read_bytes())
print('Built T090 notebook successfully')
