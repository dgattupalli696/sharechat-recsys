"""Assemble and verify the S025 full-data row-only submission CSV from T096 outputs."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.special import expit, logit

t = Path(__file__).resolve().parent
root = t.parents[1]
art = root / 'artifacts'


def blend_s025(s024, xgb_d6, q_tabm_s42, q_tabm_s43, dcnv2_r64):
    q_tabm = 0.5 * (q_tabm_s42 + q_tabm_s43)
    def lg(p): return logit(np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    z = (7.0 * lg(s024) + 2.0 * lg(xgb_d6) + 1.0 * lg(q_tabm) + 1.0 * lg(dcnv2_r64)) / 11.0
    return expit(z)


def main(folder):
    folder = Path(folder)
    sample = pl.read_csv(root / 'data/sample_submission.csv')
    ids = sample['id'].to_numpy()
    assert len(ids) == 1037090
    np.testing.assert_array_equal(ids, np.load(folder / 'ids.npy'))

    s024_df = pl.read_csv(art / 'local-t087/submission-full-rowonly-landmarks.csv')
    np.testing.assert_array_equal(ids, s024_df['id'].to_numpy())
    s024 = s024_df['watched'].to_numpy()

    def load_new(name, json_name):
        meta = json.loads((folder / f'{json_name}.json').read_text())
        assert meta['fit'] == 1537638 and meta['test'] == 1037090
        v = np.load(folder / f'{name}.npy')
        assert v.shape == (1037090,) and np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
        return v

    xgb_d6 = load_new('xgb_d6', 'xgb_d6')
    dcnv2_r64 = load_new('dcnv2_r64_s42', 'dcnv2_r64-42')
    q_tabm_42 = load_new('q_tabm_s42', 'q_tabm-42')
    q_tabm_43 = load_new('q_tabm_s43', 'q_tabm-43')

    pred = blend_s025(s024, xgb_d6, q_tabm_42, q_tabm_43, dcnv2_r64)
    assert pred.shape == (1037090,) and np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    # Singleton blend invariance check
    np.testing.assert_allclose(pred[:1], blend_s025(s024[:1], xgb_d6[:1], q_tabm_42[:1], q_tabm_43[:1], dcnv2_r64[:1]), rtol=1e-12)

    dest = art / 'local-t096/submission-full-rowonly-superblend.csv'
    dest.parent.mkdir(exist_ok=True, parents=True)
    pl.DataFrame({'id': ids, 'watched': pred}).write_csv(dest)
    check = pl.read_csv(dest)
    np.testing.assert_array_equal(ids, check['id'].to_numpy())
    np.testing.assert_allclose(check['watched'].to_numpy(), pred, rtol=0, atol=1e-15)
    rep = {
        'file': str(dest.relative_to(root)),
        'rows': len(pred),
        'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
        'mean_pred': float(pred.mean()),
    }
    (t / 'candidate.json').write_text(json.dumps(rep, indent=2) + '\n')
    print(rep)


if __name__ == '__main__':
    main(sys.argv[1])
