"""Joint phase/pulse design with exchanged remote-ambiguity constraints.

Distances use the true-point small-noise covariance. They are design
diagnostics, not global error bounds. No simulated observation is used to
select a setting in this fixed-design study.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize
from drift_study import layout
from two_beam import probability, resources, atom_counts, control_size
from two_beam import evaluate as linear_evaluate, statistics as linear_statistics
from two_beam_optimize import BASE

OUT=Path(__file__).resolve().parent/'results'


def evaluate(v,kind,pattern,config,bs,details=False,full=False):
    if config.get('quadratic_covariance',False):
        from two_beam_quadratic import evaluate as quadratic_evaluate
        return quadratic_evaluate(v,kind,pattern,config,bs,details,full)
    return linear_evaluate(v,kind,pattern,config,bs,details,full)


def statistics(v,kind,pattern,b,config,substeps=3,full=False):
    if config.get('quadratic_covariance',False):
        from two_beam_quadratic import statistics as quadratic_statistics
        return quadratic_statistics(v,kind,pattern,b,config,substeps,full)
    return linear_statistics(v,kind,pattern,b,config,substeps,full)


def means(v,kind,pattern,config,points):
    flags,phases,_,_=layout(v,pattern,config)
    points=np.asarray(points)
    return probability(points[:,0],points[:,1],
        np.zeros((len(points),len(flags),2,6)),v,kind,flags,phases,config)


def scan(v,kind,pattern,config,truth_bs=None,truth_deltas=None,grid_size=81):
    if truth_bs is None:truth_bs=np.linspace(-.6,.6,9)
    if truth_deltas is None:truth_deltas=[-.004,.0005,.004]
    remote_b,remote_d=np.meshgrid(np.linspace(-.8,.8,grid_size),
                                 np.linspace(-.01,.01,grid_size))
    points=np.column_stack([remote_d.ravel(),remote_b.ravel()])
    remote=means(v,kind,pattern,config,points)
    moments=evaluate(v,kind,pattern,config,truth_bs,True,True)['records']
    rows=[]
    for i,b in enumerate(truth_bs):
        precision=np.linalg.inv(moments[i]['covariance'])
        for delta in truth_deltas:
            truth=means(v,kind,pattern,config,[[delta,b]])[0]
            residual=remote-truth
            distances=np.einsum('ni,ij,nj->n',residual,precision,residual)
            allowed=abs(points[:,1]-b)>.15
            distances[~allowed]=np.inf
            index=int(np.argmin(distances))
            rows.append(dict(truth=[float(delta),float(b)],remote=points[index].tolist(),
                             distance_squared=float(distances[index])))
    return sorted(rows,key=lambda r:r['distance_squared'])


class JointDesign:
    def __init__(self,design,config):
        self.design=design;self.config=config;self.kind=design['family']
        self.base=np.array(design['parameters'])
        if config.get('ratio_modulation',False) and len(self.base)==control_size(self.kind)+1:
            self.base=np.r_[self.base,np.zeros(4)]
        self.flags,self.phases,_,_=layout(self.base,design['pattern'],config)
        self.groups=np.empty(len(self.flags),int)
        for flag,count,offset in [(False,6,0),(True,2,6)]:
            ids=np.flatnonzero(self.flags==flag)
            self.groups[ids]=np.arange(len(ids))%count+offset
        extra=10 if self.kind=='shape' else 6
        # Fix redundant global/short offsets; the eight phase classes are free.
        self.free=[i for i in range(len(self.base)) if i not in (5,extra+1)]
        bounds=list(BASE)+([(-np.pi,np.pi)]*4 if self.kind=='shape' else [])
        bounds[3]=(bounds[3][0],config.get('long_time_upper',bounds[3][1]))
        bounds += [(-1.,1.),(-np.pi,np.pi),(.05,.95)]
        if config.get('ratio_modulation',False):bounds += [(-.6,.6)]*4
        self.bounds=[bounds[i] for i in self.free]+[(-np.pi,np.pi)]*8
        self.initial=np.r_[self.base[self.free],np.zeros(8)]

    def unpack(self,x):
        v=self.base.copy();v[self.free]=x[:len(self.free)]
        phases=self.phases+x[len(self.free):][self.groups]
        pattern=dict(long_flags=self.flags.tolist(),phases=phases.tolist())
        return v,pattern


def optimize(design,config,cycles,maxiter,threshold):
    joint=JointDesign(design,config);best=joint.initial.copy()
    bs=list(np.linspace(-.6,.6,13));pairs=[];history=[]
    free_config=dict(config,time_cap=1e8,exposure_cap=1e8,peak_cap=1e8)
    for cycle in range(cycles):
        v,pattern=joint.unpack(best)
        found=scan(v,joint.kind,pattern,free_config)
        for row in found:
            if row['distance_squared']<threshold*1.5:
                pair=(row['truth'],row['remote'])
                if pair not in pairs:pairs.append(pair)
        truth_bs=sorted(set(bs+[p[0][1] for p in pairs]))
        b_index={b:i for i,b in enumerate(truth_bs)}
        cache={}
        def constraints(y):
            key=y.tobytes()
            if key in cache:return cache[key]
            v,pattern=joint.unpack(y[:-1])
            moments=evaluate(v,joint.kind,pattern,free_config,truth_bs,True,True)['records']
            cost=resources(v,pattern,config)
            c=[y[-1]-moments[b_index[b]]['variance']/1e-8 for b in bs]
            c.extend([(config['time_cap']-cost['elapsed'])/100,
                      (config['exposure_cap']-cost['exposure'])/10])
            width=config.get('b_width',.6);factor=np.sqrt(1+width*width)
            peak_cap=config.get('peak_cap',factor)
            if config.get('ratio_modulation',False):
                # Smooth endpoint inequalities avoid differentiating max/abs at q=0.
                q=v[-4:]
                for sign in (-1.,1.):c.extend(peak_cap-v[4]*(factor*np.cosh(q)+sign*width*np.sinh(q)))
            else:c.append(peak_cap-cost['peak'])
            if pairs:
                ps=means(v,joint.kind,pattern,config,[p for pair in pairs for p in pair])
                precision={b:np.linalg.inv(moments[b_index[b]]['covariance'])
                           for b in set(p[0][1] for p in pairs)}
                for j,pair in enumerate(pairs):
                    r=ps[2*j]-ps[2*j+1]
                    c.append((r@precision[pair[0][1]]@r-threshold)/threshold)
            cache.clear();cache[key]=np.asarray(c)
            return cache[key]
        start=np.r_[best,evaluate(v,joint.kind,pattern,free_config,bs)/1e-8]
        fit=minimize(lambda y:y[-1],start,method='SLSQP',
            bounds=joint.bounds+[(.01,1000.)],constraints=[dict(type='ineq',fun=constraints)],
            options=dict(maxiter=maxiter,ftol=2e-8,eps=2e-6))
        violation=float(min(constraints(fit.x)))
        accepted=violation>=-2e-5
        if accepted:best=fit.x[:-1]
        v,pattern=joint.unpack(best)
        dense=evaluate(v,joint.kind,pattern,free_config,np.linspace(-.6,.6,81),True)['records']
        bs=sorted(set(bs+[r['b'] for r in sorted(dense,key=lambda r:r['variance'])[-4:]]))
        audit=scan(v,joint.kind,pattern,free_config)
        record=dict(cycle=cycle,success=bool(fit.success),accepted=accepted,message=str(fit.message),
                    constraint_minimum=violation,variance=max(r['variance'] for r in dense),
                    ambiguity_minimum=audit[0],pair_count=len(pairs))
        history.append(record);print(json.dumps(record),flush=True)
    v,pattern=joint.unpack(best)
    before_projection=v.copy()
    cost=resources(v,pattern,config)
    peak_cap=config.get('peak_cap',np.sqrt(1+config.get('b_width',.6)**2))
    scale=min(1.,peak_cap/cost['peak'],config['exposure_cap']/cost['exposure'])
    if scale<1.:v[4]*=scale*(1-1e-12)
    if cost['elapsed']>config['time_cap']:
        v[3]-=(cost['elapsed']-config['time_cap'])/sum(joint.flags)+1e-10
    checked_config=dict(config,integer_atoms=True)
    finite=[statistics(v,joint.kind,pattern,b,checked_config,3) for b in np.linspace(-.6,.6,81)]
    worst=max(finite,key=lambda r:r['variance'])
    return dict(family=joint.kind,parameters=v.tolist(),pattern=pattern,readout='joint-coded',
                variance=worst['variance'],resources=resources(v,pattern,config),
                atoms_per_shot=atom_counts(v,joint.kind,joint.flags,checked_config).tolist(),
                components={k:worst[k] for k in ('shot_variance','common_variance','difference_variance','quadratic_variance') if k in worst},
                ambiguity_scan=scan(v,joint.kind,pattern,checked_config,grid_size=121),
                feasibility_projection_max_change=float(np.max(abs(v-before_projection))),
                search_history=history)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--families',nargs='+',default=['hyper','shape'])
    parser.add_argument('--cycles',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--threshold',type=float,default=36.)
    parser.add_argument('--scan-only',action='store_true')
    parser.add_argument('--quadratic',action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--output',default='two_beam_joint_coded.json')
    args=parser.parse_args()
    rows=[]
    for kind in args.families:
        name=args.source or ('two_beam_two-group_allocated.json' if kind=='hyper' else 'two_beam_discrimination.json')
        source=json.loads((OUT/name).read_text())
        config=dict(source['config'],integer_atoms=False,quadratic_covariance=args.quadratic)
        design=min((r for r in source['results'] if r['family']==kind),key=lambda r:r['variance'])
        started=time.monotonic()
        if args.scan_only:
            audit=scan(design['parameters'],kind,design['pattern'],config)
            row=dict(family=kind,scan=audit)
        else:row=optimize(design,config,args.cycles,args.iterations,args.threshold)
        row['elapsed_seconds']=time.monotonic()-started
        rows.append(row)
        print(json.dumps(dict(family=kind,elapsed_seconds=row['elapsed_seconds'],
                             minimum=(row.get('scan') or row['ambiguity_scan'])[0])),flush=True)
        (OUT/args.output).write_text(json.dumps(dict(config=config,threshold=args.threshold,results=rows,
            scope='Fixed phase classes with pulse controls; finite-grid truth-covariance distances are not global bounds.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
