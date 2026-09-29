"""Verify calibration value in the explicitly local Gaussian scalar model."""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parent / 'results'


def main():
    rng = np.random.default_rng(187335)
    reps = 300000
    a, vb, sd, readout, tau = 1.1, .0006, .00045, .0002, 300.
    rows = []
    for delay in (0., 100., 300., 900., 3000.):
        r = np.exp(-delay/tau)
        b = rng.normal(0, np.sqrt(vb), reps)
        dc = rng.normal(0, np.sqrt(sd), reps)
        dm = r*dc + np.sqrt((1-r*r)*sd)*rng.normal(size=reps)
        z = b-a*dc+rng.normal(0, np.sqrt(readout), reps)
        target = b-a*dm
        covariance = vb+a*a*r*sd
        vz = vb+a*a*sd+readout
        prior = vb+a*a*sd
        prediction = covariance/vz*z
        theory = prior-covariance**2/vz
        errors = (prediction-target)**2
        standard_error = errors.std(ddof=1)/np.sqrt(reps)
        assert abs(errors.mean()-theory) < 5*standard_error
        raw_theory = readout+2*a*a*sd*(1-r)
        raw_errors = (z-target)**2
        raw_se = raw_errors.std(ddof=1)/np.sqrt(reps)
        assert abs(raw_errors.mean()-raw_theory) < 5*raw_se
        rows.append(dict(delay=delay, correlation=r, posterior_variance=theory,
                         monte_carlo_mse=errors.mean(), monte_carlo_se=standard_error,
                         unpredicted_reuse_variance=raw_theory,
                         information_value=prior-theory))
    assert all(rows[i]['posterior_variance'] <= rows[i+1]['posterior_variance'] for i in range(len(rows)-1))
    result = dict(parameters=dict(a=a, prior_b_variance=vb, differential_ou_variance=sd,
                                 readout_variance=readout, correlation_time=tau),
                  repetitions=reps, results=rows,
                  scope='Local frozen sensitivity and Gaussian prior; verifies the scalar derivation, not the nonlinear Raman risk or a quantum precision bound.')
    (OUT / 'calibration_value_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
