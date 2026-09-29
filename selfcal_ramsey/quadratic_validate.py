"""Independent Hessian contraction and estimator-constraint checks."""
from pathlib import Path
import json
import numpy as np
from two_beam_quadratic import corrections, statistics
from two_beam_local_audit import quadratic_terms
from two_beam_inference import second_order_mean


def main():
    out=Path(__file__).resolve().parent/'results'
    source=json.loads((out/'two_beam_joint_coded.json').read_text())
    config=dict(source['config'],integer_atoms=True);rows=[]
    for design in source['results']:
        v=design['parameters'];kind=design['family'];pattern=design['pattern']
        bs=np.array([-.6,0.,.6])
        shifts,covs=corrections(v,kind,pattern,config,bs)
        half_shift,half_cov=corrections(v,kind,pattern,dict(config,sigma_intensity=config['sigma_intensity']/2),bs)
        np.testing.assert_allclose(half_shift*4,shifts,atol=1e-14)
        np.testing.assert_allclose(half_cov*16,covs,atol=1e-14)
        other_mean=second_order_mean(design,config,bs)
        assert np.max(abs(other_mean-shifts))<1e-6
        for j,b in enumerate(bs):
            scalar_shift,scalar_cov=quadratic_terms(design,config,b,step=.001)
            np.testing.assert_allclose(shifts[j],scalar_shift,atol=1e-12)
            np.testing.assert_allclose(covs[j],scalar_cov,atol=1e-12)
            row=statistics(v,kind,pattern,b,config,3,True)
            np.testing.assert_allclose(row['weights']@row['jacobian'],[1.,0.],atol=1e-8)
            np.testing.assert_allclose(row['weights']@row['covariance']@row['weights'],row['variance'],rtol=1e-8)
            parts=sum(row[k] for k in ['shot_variance','common_variance','difference_variance','quadratic_variance'])
            np.testing.assert_allclose(parts,row['variance'],rtol=1e-8)
            assert np.linalg.eigvalsh(row['covariance']).min()>0
            rows.append(dict(family=kind,b=float(b),variance=row['variance'],
                minimum_quadratic_eigenvalue=float(np.linalg.eigvalsh(covs[j]).min()),
                independent_mean_error=float(np.max(abs(other_mean[j]-shifts[j])))))
    (out/'quadratic_validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
