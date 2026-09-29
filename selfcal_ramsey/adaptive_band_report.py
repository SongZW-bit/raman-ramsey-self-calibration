"""Report independent confirmation runs without selecting a favorable baseline."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / 'results'
POINTS = [-.6, -.5, -.4, .4, .5, .6]


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def extract(data, key, value, mse_key):
    rows = sorted((r for r in data['results'] if r[key] == value), key=lambda r: r['b'])
    assert [r['b'] for r in rows] == POINTS, 'Confirmation run is incomplete'
    assert all(r['reps'] == 4000 for r in rows)
    return np.array([r[mse_key] for r in rows]), np.array([r['mse_standard_error'] for r in rows])


def comparison(candidate, reference, count=1):
    m, s = candidate
    r, t = reference
    ratio = m / r
    error = np.sqrt((s / r)**2 + (m * t / r**2)**2)
    z = norm.ppf(1 - .05 / (2 * count))
    return dict(mse_reduction=1-ratio, rmse_reduction=1-np.sqrt(ratio),
                mse_reduction_standard_error=error,
                mse_reduction_interval=[1-ratio-z*error, 1-ratio+z*error])


def serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def main():
    fixed = read('two_beam_fixed_band_mc.json')
    frequency = read('two_beam_adaptive_cal2_band.json')
    phase = read('two_beam_adaptive_phase_band.json')
    assert frequency['true_delta'] == phase['true_delta'] == .0005
    assert all(r['true_delta'] == phase['true_delta'] for r in fixed['results'])
    assert frequency['calibration_shots'] == phase['calibration_shots'] == 2
    assert frequency['resources'] == phase['resources']
    assert frequency['prefix_and_resource_checks'] and phase['prefix_and_resource_checks']
    config = phase['config']
    physics = ['sigma_intensity', 'beam_correlation', 'correlation_time', 'gamma', 'total_atoms']
    for design in fixed['designs']:
        for key in physics:
            assert design['config'].get(key) == config.get(key), key
    series = {
        'fixed_hyper': extract(fixed, 'family', 'hyper', 'delta_mse'),
        'fixed_shape': extract(fixed, 'family', 'shape', 'delta_mse'),
        'adaptive_frequency': extract(frequency, 'mode', 'adaptive', 'mse'),
        'adaptive_frequency_phase': extract(phase, 'mode', 'adaptive', 'mse'),
    }
    rows = []
    for i, b in enumerate(POINTS):
        candidate = tuple(x[i] for x in series['adaptive_frequency_phase'])
        row = dict(b=b, mse={name: values[0][i] for name, values in series.items()})
        for name in ('fixed_hyper', 'fixed_shape', 'adaptive_frequency'):
            row['versus_'+name] = comparison(candidate, tuple(x[i] for x in series[name]))
        row['versus_shape_six_comparison_interval'] = comparison(
            candidate, tuple(x[i] for x in series['fixed_shape']), 6)['mse_reduction_interval']
        rows.append(row)
    # Equal weighting describes these six tested points, not a continuous prior.
    means = {name: (m.mean(), np.linalg.norm(s)/len(POINTS)) for name, (m, s) in series.items()}
    averages = {name: comparison(means['adaptive_frequency_phase'], means[name])
                for name in ('fixed_hyper', 'fixed_shape', 'adaptive_frequency')}
    resources = {d['design']['family']: d['design']['resources'] for d in fixed['designs']}
    resources['adaptive'] = phase['resources']
    result = dict(points=rows, six_point_mean_comparisons=averages, resources=resources,
                  true_delta=phase['true_delta'], repetitions_per_point=4000,
                  sources=['two_beam_fixed_band_mc.json', 'two_beam_adaptive_cal2_band.json',
                           'two_beam_adaptive_phase_band.json'],
                  interval_method='Independent-run delta method, normal approximation. Pointwise 95%; six-comparison intervals use Bonferroni. No model or optimizer uncertainty.',
                  scope='Frozen controls after exploratory pilots. Six selected confirmation points at one frequency and one drift model; no continuous-band or literature-wide optimum claim. Shape is the designated main fixed comparator; hyper remains visible.')
    (OUT / 'adaptive_band_summary.json').write_text(json.dumps(result, indent=2, default=serializable), encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    x = np.arange(len(POINTS))
    styles = [('fixed_hyper', 'Fixed Hyper-Ramsey family', '#58616b', 's'),
              ('fixed_shape', 'Fixed phase-shaped', '#b16b19', 'D'),
              ('adaptive_frequency', 'Adaptive frequency', '#27855b', '^'),
              ('adaptive_frequency_phase', 'Adaptive frequency + phase', '#1765ac', 'o')]
    for k, (name, label, color, marker) in enumerate(styles):
        m, s = series[name]
        axes[0].errorbar(x+(k-1.5)*.15, m/1e-8, yerr=1.96*s/1e-8, color=color,
                         marker=marker, ms=4, linestyle='none', capsize=2, label=label)
    gain = [r['versus_fixed_shape']['mse_reduction']*100 for r in rows]
    error = [r['versus_fixed_shape']['mse_reduction_standard_error']*196 for r in rows]
    axes[1].errorbar(x, gain, yerr=error, color='#1765ac', fmt='o', capsize=4)
    axes[1].axhline(0, color='#555555', lw=.8)
    axes[0].set_ylabel('Frequency MSE (normalized units, $10^{-8}$)')
    axes[1].set_ylabel('MSE reduction vs fixed phase-shaped (%)')
    axes[0].set_title('Independent Monte Carlo confirmation', fontsize=12, loc='left')
    axes[1].set_title('Advantage depends on operating point', fontsize=12, loc='left')
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    axes[0].set_ylim(1.45, 2.95)
    axes[1].set_ylim(-1, 31)
    for ax in axes:
        ax.set_xticks(x, [f'{b:+.1f}' for b in POINTS])
        ax.set_xlabel('Unknown static light-shift parameter $b$ (discrete tests)')
        ax.grid(axis='y', alpha=.18)
        ax.axvline(2.5, color='#bbbbbb', ls=':', lw=1)
    fig.suptitle('Two initial calibrations; equal resource caps; 4,000 records per point', fontsize=12)
    fig.savefig(OUT / 'adaptive_band_confirmation.png', dpi=190)
    fig.savefig(OUT / 'adaptive_band_confirmation.pdf')
    print(json.dumps(result, indent=2, default=serializable))


if __name__ == '__main__':
    main()
