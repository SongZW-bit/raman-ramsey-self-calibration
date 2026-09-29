"""Check saved sampling completeness and independently reconstruct risk totals."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OLD = ROOT.parent / 'results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_rows(rows, cells, reps, arms, decay_resolution):
    assert len(rows) == cells * len(arms)
    grouped = {}
    for r in rows:
        key = (r['b'], r['delta'], r['seed'])
        grouped.setdefault(key, {})[r['name']] = r
        assert r['name'] in arms and r['reps'] == reps
        assert r['substeps'] == 1 and not r['effective']
        x = np.asarray(r['estimates'])
        y = np.asarray(r['data'])
        assert x.shape == (reps, 2) and y.shape == (reps, 12)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert (y >= 0).all() and (y <= 1).all()
        assert np.isclose(np.mean((x[:, 0] - r['delta'])**2), r['mse'],
                          rtol=1e-12, atol=1e-20)
        choices = r['choices']
        if r['name'].startswith('adaptive'):
            q = np.asarray(choices)
            assert q.shape == (reps,) and (q >= 0).all() and (q < 8).all()
        else:
            assert choices is None
    assert len(grouped) == cells
    for group in grouped.values():
        assert set(group) == set(arms)
        if 'adaptive8_mc' in group:
            assert np.array_equal(np.asarray(group['adaptive8']['data'])[:, :2],
                                  np.asarray(group['adaptive8_mc']['data'])[:, :2])
    return dict(cells=cells, records_per_arm=cells * reps,
                arms=list(arms), decay_resolution=decay_resolution, passed=True)


def main():
    report = {'suites': {}, 'primary': {}, 'controls': {}}
    plan = {'broad': (21, 200, 5), 'd1': (6, 64, 3),
            'ablations': (2, 500, 4), 'stress': (18, 240, 2),
            'd1_stress': (4, 64, 2), 'bank8': (7, 400, 4),
            'bank8_d1': (5, 96, 2)}
    for name, (cells, reps, arms) in plan.items():
        s = read(DATA / (name + '.json'))
        assert s['completed_cells'] == cells
        assert len(s['rows']) == cells * arms
        keys = set()
        for r in s['rows']:
            key = (r['seed'], r['name'])
            assert key not in keys, (name, key)
            keys.add(key)
            x, y = np.asarray(r['estimates']), np.asarray(r['data'])
            assert x.shape == (reps, 2) and y.shape == (reps, 12)
            assert r['reps'] == reps and np.isfinite(x).all() and np.isfinite(y).all()
            assert np.min(y) >= 0 and np.max(y) <= 1
            assert np.max(abs(x[:, 0])) <= .003500001
            assert np.max(abs(x[:, 1])) <= .8000001
            risk = np.mean((x[:, 0] - r['delta']) ** 2)
            assert np.isclose(risk, r['mse'], rtol=1e-12, atol=1e-20)
            if r['choices'] is not None:
                choices = np.asarray(r['choices'])
                assert choices.shape == (reps,)
                assert np.min(choices) >= 0
                assert np.max(choices) < (8 if r['name'] == 'adaptive8' else 3)
        report['suites'][name] = dict(cells=cells, arms=arms,
                                     records_per_arm=cells*reps, passed=True)
    for model in ('effective', 'd1'):
        s = read(DATA / ('convergence_' + model + '.json'))
        assert len(s['rows']) == 8 and s['reps'] == 96
        for r in s['rows']:
            x = np.array(r['estimates'])
            assert x.shape == (96, 2) and np.isfinite(x).all()
        m = read(DATA / ('moments_' + model + '.json'))
        assert len(m['rows']) == 5
        assert m['independent_paths'] == (12000 if model == 'effective' else 192)
    means = {}
    for arm in ('fixed_shape', 'adaptive'):
        losses = []
        for side in ('negative', 'positive'):
            s = read(OLD / f'rb87_effective_fine_{side}1600.json')
            row = next(r for r in s['rows'] if r['name'] == arm)
            x = np.asarray(row['estimates'])
            assert len(x) == 1600 and row['substeps'] == 3
            losses.extend((x[:, 0] - s['true_delta'])**2)
        means[arm] = float(np.mean(losses))
    summary = read(OLD / 'rb87_effective_fine_summary.json')
    for arm, value in means.items():
        assert np.isclose(value, summary['means'][arm], rtol=1e-12, atol=1e-20)
    report['primary']['effective'] = dict(risks=means, gain=1-means['adaptive']/means['fixed_shape'])
    coarse = read(OLD / 'rb87_refined_d1_mc200.json')
    means = {}
    for arm in ('fixed_shape', 'adaptive'):
        coarse_losses, corrections = [], []
        for side, b in (('negative', -.6), ('positive', .6)):
            row = next(r for r in coarse['rows'] if r['name'] == arm and r['b'] == b)
            coarse_losses.extend((np.array(row['estimates'])[:, 0]-coarse['true_delta'])**2)
            s = read(OLD / f'rb87_resolution_{side}64.json')
            assert s['seed'] != coarse['seed']
            pairs = {r['substeps']: (np.array(r['estimates'])[:, 0]-s['true_delta'])**2
                     for r in s['rows'] if r['name'] == arm}
            assert all(len(v) == 64 for v in pairs.values())
            corrections.extend(pairs[3]-pairs[1])
        means[arm] = float(np.mean(coarse_losses)+np.mean(corrections))
    summary = read(OLD / 'rb87_resolution_d1_summary.json')
    for arm, value in means.items():
        assert np.isclose(value, summary['corrected_risks'][arm], rtol=1e-12, atol=1e-20)
    report['primary']['d1'] = dict(risks=means, gain=1-means['adaptive']/means['fixed_shape'])
    for row in read(DATA / 'controls_complete.json'):
        counts = np.asarray(row['counts'])
        assert counts.shape == (12,) and counts.sum() == 160000
        assert np.all(counts > 0) and np.all(counts == counts.astype(int))
        assert row['resources']['atoms'] == 160000
        report['controls'][row['name']] = {'atoms': int(counts.sum()), 'passed': True}
    assert len(read(DATA / 'model_mismatch.json')['rows']) == 36
    assert read(OLD / 'rb87_multilevel_validation.json')['passed']
    new_plan = {
        'primary_holdout': (24, 96), 'primary_broad': (27, 96),
        'resolved_d1_holdout': (24, 64), 'resolved_d1_endpoints': (2, 160),
    }
    for name, (cells, reps) in new_plan.items():
        s = read(DATA / (name + '.json'))
        assert s['completed_cells'] == cells
        assert len(s['cells']) == cells
        assert s['decay_resolution'] == ('excited_hyperfine' if name.startswith('resolved')
                                         else 'unresolved_excited')
        arms = ('fixed_shape', 'fixed_refined', 'adaptive8', 'adaptive8_mc')
        if name == 'resolved_d1_endpoints':
            arms = ('fixed_shape', 'fixed_refined', 'adaptive8')
        report['suites'][name] = check_rows(s['rows'], cells, reps, arms,
                                             s['decay_resolution'])
    o = read(DATA / 'primary_oracle.json')
    assert o['completed_cells'] == o['planned_cells'] == len(o['rows']) == 7
    for r in o['rows']:
        losses = np.asarray(r['losses'])
        assert losses.shape == (96, 8) and np.isfinite(losses).all()
        assert (losses >= 0).all()
        for name in ('choices_local', 'choices_mc'):
            choices = np.asarray(r[name])
            assert choices.shape == (96,) and (choices >= 0).all() and (choices < 8).all()
    report['suites']['primary_oracle'] = dict(cells=7, records=672, passed=True)
    training = np.load(DATA / 'primary_selector_mc.npz')
    assert training['ycal'].shape == (3000, 2)
    assert training['losses'].shape == (3000, 8)
    assert training['b'].shape == training['delta'].shape == (3000,)
    assert np.isfinite(training['losses']).all() and (training['losses'] >= 0).all()
    report['suites']['conditional_selector'] = dict(training_records=3000, passed=True)
    impurity = read(DATA / 'impurity_mc.json')
    assert impurity['decay_resolution'] == 'unresolved_excited'
    assert len(impurity['rows']) == 6 and impurity['reps_per_cell'] == 64
    for r in impurity['rows']:
        assert len(r['scenarios']) == 3
        for scenario in r['scenarios']:
            loss = np.asarray(scenario['loss'])
            assert loss.shape == (64,) and np.isfinite(loss).all() and (loss >= 0).all()
    report['suites']['impurity_mc'] = dict(cells=6, records_per_scenario=384, passed=True)
    for filename, arms in [('resolved_200_mc.json', 2),
                           ('resolved_200_square_mc.json', 1)]:
        s = read(DATA / filename)
        assert len(s['rows']) == 3 * arms
        for row in s['rows']:
            estimates = np.asarray(row['estimates'])
            assert estimates.shape == (200, 2) and np.isfinite(estimates).all()
            assert np.isclose(np.mean((estimates[:, 0] - s['true_delta'])**2),
                              row['mse'], rtol=1e-12, atol=1e-20)
        report['suites'][filename] = dict(cells=len(s['rows']),
                                           records_per_cell=200, passed=True)
    dissipator = read(DATA / 'dissipator_audit.json')
    assert len(dissipator) == 18
    target_difference = 0.0
    for row in dissipator:
        for grouping in ('resolved', 'nonsecular'):
            state = row[grouping]
            assert abs(state['trace'] - 1) < 1e-7
            assert state['min_eigenvalue'] > -1e-8
            assert 0 <= state['target'] <= 1
            assert 0 <= state['spectator'] <= 1
            assert state['emitted_photons'] >= 0
        target_difference = max(target_difference,
                                abs(row['resolved']['target'] - row['nonsecular']['target']))
    paths = read(DATA / 'scattering_interference.json')
    assert len(paths) == 4
    for row in paths:
        assert np.isclose(row['coherent_sum'], row['separate_sum'], rtol=1e-10)
        assert all(x['coherent'] >= 0 and x['separate'] >= 0
                   for x in row['channels'])
    report['suites']['dissipator'] = dict(cases=18,
                                           maximum_target_difference=target_difference,
                                           scattering_sums_agree=True, passed=True)
    output = ROOT / 'audit'
    output.mkdir(exist_ok=True)
    (output / 'data_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
