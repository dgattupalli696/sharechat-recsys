import json
from pathlib import Path

t = Path(__file__).resolve().parent
t014_nb = json.loads((t.parent / '014-full-gpu-combinations-architectures/notebook.ipynb').read_text())
wheels_cell = dict(t014_nb['cells'][2])
wheels_cell['source'] = ['import sys\n', 'from pathlib import Path\n'] + list(wheels_cell['source'])

b92 = (t.parent / '092-super-enriched-trees/worker.py').read_text()
b94 = (t.parent / '094-neural-diversity-and-rank/worker.py').read_text()
q81 = (t.parent / '081-row-only-quantile-neural/quantiles.py').read_text()
w95 = (t / 'worker.py').read_text()
for n, s in [('t092', b92), ('t094', b94), ('q81', q81), ('t095', w95)]:
    compile(s, n, 'exec')

run_code = (
    "from pathlib import Path\nimport subprocess, sys\n"
    f"Path('/kaggle/working/t092_worker.py').write_text({b92!r})\n"
    f"Path('/kaggle/working/t094_worker.py').write_text({b94!r})\n"
    f"Path('/kaggle/working/quantiles81.py').write_text({q81!r})\n"
    f"Path('/kaggle/working/t095_worker.py').write_text({w95!r})\n"
    "stages = [['prepare']] + [\n"
    "    [fold, arch, str(seed)]\n"
    "    for fold in ['earlier', 'primary']\n"
    "    for arch, seed in [('q_tabm', 42), ('q_tabm', 43), ('q_moe4', 42), ('q_moe4', 43)]\n"
    "]\n"
    "for args in stages:\n"
    "    print('=== STAGE', args, flush=True)\n"
    "    subprocess.run([sys.executable, '-u', '/kaggle/working/t095_worker.py', *args], check=True)\n"
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
print('Built T095 notebook successfully')
