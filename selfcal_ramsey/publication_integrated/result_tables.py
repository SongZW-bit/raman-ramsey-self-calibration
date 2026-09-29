"""Create supplemental tables directly from audited JSON records."""
import json
from pathlib import Path
import numpy as np
from make_figures import summarize, LABEL
from study import ROOT, DATA


TABLE_LABELS=iter(['comparisons','stress_all','convergence','moments','mismatch',
                   'expanded'])


def table(caption,headers,rows,align):
    return '\n'.join([r'\begin{table*}[t]',r'\caption{'+caption+'}',r'\label{tab:'+next(TABLE_LABELS)+'}',r'\begin{ruledtabular}',r'\begin{tabular}{'+align+'}', ' & '.join(headers)+r' \\',r'\hline',*[' & '.join(map(str,r))+r' \\' for r in rows],r'\end{tabular}',r'\end{ruledtabular}',r'\end{table*}'])


def main():
    summary=summarize();output=[]
    labels={'broad':'Effective broad','bank8':'Effective bank coverage','bank8_d1':'D1 bank coverage','d1':'D1 interior','ablations':'Effective ablation'}
    combined=[]
    for key in labels:
        if key not in summary:continue
        rows=[]
        for r in summary[key]['comparisons']:
            rows.append([LABEL[r['reference']],LABEL[r['adaptive']],f"{1e8*r['fixed_mse']:.4f}",f"{1e8*r['adaptive_mse']:.4f}",f"{100*r['gain']:.2f}",f"[{100*r['ci'][0]:.2f}, {100*r['ci'][1]:.2f}]"])
        combined.extend([[labels[key]]+row for row in rows])
    output.append(table('Finite-count comparisons. Risks are in Hz$^2$; gains and paired 95\\% bootstrap intervals are in percent. Each parameter cell receives equal weight. Three- and eight-branch selection are denoted Adaptive (3) and Adaptive (8).', ['Ensemble','Reference','Compared','Ref. risk','Risk','Gain',r'95\% interval'],combined,'lllrrrr'))
    combined=[]
    for key in ['stress','d1_stress']:
        if key not in summary:continue
        rows=[]
        for r in summary[key]:
            rows.append([f"{r['tau']:g}",f"{r['sigma']:g}",f"{r['rho']:g}",f"{r['delay']:g}",f"{1e8*r['fixed_mse']:.3f}",f"{1e8*r['adaptive_mse']:.3f}",f"{100*r['gain']:.2f}",f"[{100*r['ci'][0]:.2f}, {100*r['ci'][1]:.2f}]",f"{100*r['time_weighted_gain']:.2f}"])
        combined.extend([[('Effective' if key=='stress' else 'D1')]+row for row in rows])
    output.append(table('Noise and latency tests with nominal inference. Risks are Hz$^2$; gains and intervals are percent. The final column gives the reduction in risk times actual block duration, including the adaptive delay.', ['Model',r'$\tau$',r'$\sigma_I$',r'$\rho_I$',r'$L$','Ref. risk','Risk','Gain',r'95\% interval',r'$G_t$'],combined,'lrrrrrrrrr'))
    rows=[]
    for key,label in [('convergence_effective','Effective'),('convergence_d1','D1')]:
        for r in summary.get(key,[]):
            rows.append([label,LABEL[r['arm']],f"{r['b']:g}",f"{100*r['relative_risk_change']:.3f}",f"{1e4*r['estimate_difference_rms']:.4f}", '--' if r['branch_switches'] is None else str(r['branch_switches'])])
    output.append(table('Temporal refinement from 3 to 9 OU samples per phase segment, with 96 paired records per endpoint and protocol. The RMS estimate difference is in Hz. Adaptive decisions are recomputed. Relative risk changes refer to this independent refinement ensemble.', ['Model','Protocol',r'$b$',r'$\Delta R/R$ (\%)','RMS difference','Switches'],rows,'llrrrr'))
    rows=[]
    for name,label in [('moments_effective','Effective'),('moments_d1','D1')]:
        p=DATA/(name+'.json')
        if not p.exists():continue
        for r in json.loads(p.read_text())['rows']:
            rows.append([label,r['name'].replace('_',r'\_'),f"{r['b']:g}",f"{r['target_bias_over_sd']:.4f}",f"{r['target_variance_ratio']:.4f}",f"{r['drift_projection_skew']:.3f}"])
    output.append(table('Independent nonlinear-moment diagnostics, using 12000 paths per effective-model configuration and 192 per D1 configuration. Mean errors are projected onto the local target estimator and divided by its predicted standard deviation. The variance ratio is empirical nonlinear covariance divided by the inference covariance in that projection. D1 ratios carry appreciable finite-path sampling uncertainty.', ['Model','Control',r'$b$','Mean/SD','Variance ratio','Drift skew'],rows,'llrrrr'))
    p=DATA/'model_mismatch.json'
    if p.exists():
        data=json.loads(p.read_text())['rows'];rows=[]
        for source in ['d1.json','d1_stress.json','bank8_d1.json']:
            for arm in sorted({r['arm'] for r in data if r['source']==source}):
                rr=[r for r in data if r['source']==source and r['arm']==arm]
                rows.append([source.replace('.json','').replace('_',r'\_'),LABEL[arm],sum(r['reps'] for r in rr),f"{np.mean([r['matched_mse'] for r in rr])*1e8:.3f}",f"{np.mean([r['mismatched_mse'] for r in rr])*1e8:.3f}",sum(r['target_bound_hits'] for r in rr)])
        output.append(table('Likelihood transfer: effective-qubit likelihoods fitted to D1 counts with the realized D1 branch choices held fixed. Risks are in Hz$^2$. The refined fixed controls are identical in the two model lookups. Bound hits count frequency estimates at the search boundary.', ['Suite','Protocol','Records','Matched','Mismatched','Bound hits'],rows,'llrrrr'))
    primary_path=DATA/'primary_summary.json'
    resolved_path=DATA/'revision_summary.json'
    if primary_path.exists() and resolved_path.exists():
        primary=json.loads(primary_path.read_text())
        resolved=json.loads(resolved_path.read_text())
        groups=[('D1 continuous',primary.get('holdout')),
                ('D1 grid',primary.get('broad')),
                (r'D1 independent $F^\prime$ continuous',
                 resolved.get('resolved_d1_holdout.json')),
                (r'D1 independent $F^\prime$ endpoints',
                 resolved.get('resolved_d1_endpoints.json'))]
        rows=[]
        for group,source in groups:
            if not source:continue
            for r in source['comparisons']:
                rows.append([group,LABEL.get(r['reference'],r['reference']),
                             LABEL.get(r['adaptive'],r['adaptive'].replace('_',r'\_')),
                             f"{r['reference_mse_hz2']:.3f}",
                             f"{r['adaptive_mse_hz2']:.3f}",
                             f"{100*r['gain']:.1f}",
                             f"[{100*r['ci'][0]:.1f},{100*r['ci'][1]:.1f}]"])
        if rows:
            output.append(table('Expanded D1 finite-count tests. Risks are Hz$^2$; gains and 95\\% paired bootstrap intervals are percent. Continuous-domain intervals resample parameter points and records within points; fixed-grid and endpoint intervals resample records within cells. The two collapse-operator models are reported separately.',
                                ['Ensemble','Reference','Compared','Ref. risk','Risk','Gain',r'95\% interval'],rows,'lllrrrr'))
    (ROOT/'results_tables.tex').write_text('\n\n'.join(output),encoding='utf-8')
    print(json.dumps({k:summary[k].get('cells') for k in labels if k in summary}))


if __name__=='__main__':main()
