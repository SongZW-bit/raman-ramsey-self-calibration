"""Screen earlier two-phase pulse families at the new correlation time."""
from pathlib import Path
import json
import numpy as np
from drift_continuous import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    source=json.loads((OUT/'drift_refined_wide.json').read_text())
    config=dict(source['config'],correlation_time=1000.)
    rows=[]
    for previous in source['results']:
        checks=[statistics(previous['parameters'],previous['family'],previous['pattern'],b,config,3)
                for b in np.linspace(-.6,.6,161)]
        worst=max(checks,key=lambda z:z['variance'])
        row=dict(family=previous['family'],pattern=previous['pattern'],parameters=previous['parameters'],
                 variance=worst['variance'],resources=worst['resources'],
                 label=previous['family']+' '+previous['pattern'])
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='parameters'}),flush=True)
    (OUT/'drift_existing_family_audit.json').write_text(json.dumps(dict(config=config,results=rows,
        scope='Frozen earlier two-phase designs; not reoptimized at tau=1000. Hyper family is not a complete published servo.'),indent=2),encoding='utf-8')
    best=min((r for r in rows if r['family']=='hyper'),key=lambda r:r['variance'])
    (OUT/'drift_hyper_tau1000.json').write_text(json.dumps(dict(config=config,results=[best]),indent=2),encoding='utf-8')


if __name__=='__main__':main()
