from pathlib import Path
import json
t=Path(__file__).resolve().parent;prev=t.parent;net=(prev/'081-row-only-quantile-neural/worker.py').read_text();a=net.index('def fit(');b=net.index('    # Fit a robust',a);net=net[:a]+'''def fit(kind):
    from worker87 import N,V,CACHE,save
    assert kind in ('quantile','gated');torch.manual_seed(42);rng=np.random.default_rng(42);torch.set_num_threads(2);torch.set_float32_matmul_precision('high');device='cuda';y=np.load(CACHE/'y.npy');x=np.load(CACHE/'x.npy',mmap_mode='r');tr=np.arange(N);assert x.shape==(N+V,1141) and len(y)==N
'''+net[b:];net=net.replace("q=transform(x[rows],knots);return q if kind=='quantile' else .5*robust+2*q","return transform(x[rows],knots) if kind=='quantile' else robust").replace('train=cache(tr);valid=cache(va);','train=cache(tr);');net=net[:net.index('        if epoch in (2,5,8):')]+'''    model.eval()
    with torch.inference_mode():
        def pred(i,j):return model(torch.from_numpy(encode(slice(i,j))).to(device,dtype=torch.float16).float()).sigmoid().cpu().numpy()
        p=np.concatenate([pred(i,min(len(x),i+2048)) for i in range(N,len(x),2048)]);np.testing.assert_allclose(p[:12],np.concatenate([pred(i,i+1) for i in range(N,N+12)]),atol=2e-4,rtol=2e-4);save(kind,p)
''';compile(net,'neural87','exec');(t/'neural.py').write_text(net)
files=[('base62.py','062-row-only-gpu-replays/worker.py'),('rich63.py','063-row-only-embedding-depth/worker.py'),('t069.py','069-row-only-duration-budget/worker.py'),('quantiles81.py','081-row-only-quantile-neural/quantiles.py'),('rot83.py','083-row-only-paired-rotations/rotations.py'),('land86.py','086-row-only-match-landmarks/landmarks.py')];bodies=[(k,(prev/p).read_text()) for k,p in files]+[('neural87.py',net),('worker87.py',(t/'worker.py').read_text())];code='from pathlib import Path\nimport subprocess,sys,os\n';
for k,body in bodies:compile(body,k,'exec');code+=f"Path('/kaggle/working/{k}').write_text({body!r})\n"
code+="env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3')\nfor stage in ['prepare','duration','compact','paired','learned','random','gated','quantile']:\n    subprocess.run([sys.executable,'-u','/kaggle/working/worker87.py',stage],env=env,check=True)\n";prior=prev/'086-row-only-match-landmarks';nb=json.loads((prior/'notebook.ipynb').read_text());nb['cells'][0]['source']=code.splitlines(True);(t/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(t/'kernel-metadata.json').write_bytes((prior/'kernel-metadata.json').read_bytes())
