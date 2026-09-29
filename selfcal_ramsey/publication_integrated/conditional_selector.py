"""Finite-count conditional-risk surrogate for the frozen eight-branch bank."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from revision_study import estimators, DATA


FILE = DATA / "conditional_selector_mc.npz"


def train(samples=3000, seed=259253,primary=False):
    if primary:
        from complete_study import primary_estimators
        est=primary_estimators()
    else:
        est = estimators()
    branches = [est[f"branch_{j}"] for j in range(8)]
    rng = np.random.default_rng(seed)
    b = rng.uniform(-.6,.6,samples)
    delta = rng.uniform(-.002,.002,samples)
    prefix = branches[0]
    mean = prefix.mean(delta,b)[:,:2]
    cov = prefix.cov(b)[:,:2,:2]
    ycal = mean + np.einsum("nij,nj->ni",np.linalg.cholesky(cov),rng.normal(size=(samples,2)))
    losses = np.empty((samples,8))
    for j,e in enumerate(branches):
        mu = e.mean(delta,b)
        cc = e.cov(b)
        cross = cc[:,2:,:2]
        response = cross @ np.linalg.inv(cc[:,:2,:2])
        conditional_mean = mu[:,2:] + np.einsum("nij,nj->ni",response,ycal-mu[:,:2])
        conditional_cov = cc[:,2:,2:] - response @ np.swapaxes(cross,1,2)
        conditional_cov = (conditional_cov+np.swapaxes(conditional_cov,1,2))/2
        draw = conditional_mean + np.einsum("nij,nj->ni",np.linalg.cholesky(conditional_cov),
                                             rng.normal(size=(samples,cc.shape[1]-2)))
        fits = e.fit(np.column_stack((ycal,draw)))
        losses[:,j] = (fits[:,0]-delta)**2
        print(json.dumps(dict(branch=j,mean_mse_hz2=float(losses[:,j].mean()*1e8))),flush=True)
    scale = np.linalg.cholesky(np.cov(ycal.T))
    np.savez_compressed(DATA/'primary_selector_mc.npz' if primary else FILE,ycal=ycal,losses=losses,scale=scale,b=b,delta=delta,seed=seed)


class Selector:
    def __init__(self,path=None):
        s=np.load(FILE if path is None else path)
        self.losses=s["losses"]
        self.scale=s["scale"]
        self.tree=cKDTree(np.linalg.solve(self.scale,s["ycal"].T).T)

    def predict(self,ycal,k=200):
        distance,index=self.tree.query(np.linalg.solve(self.scale,np.asarray(ycal).T).T,k=k)
        bandwidth=np.maximum(distance[:,-1:],1e-8)
        weights=np.exp(-.5*(distance/bandwidth)**2)
        return np.einsum("nk,nkj->nj",weights,self.losses[index])/weights.sum(axis=1,keepdims=True)

    def select(self,ycal):
        return np.argmin(self.predict(ycal),axis=1)


if __name__ == "__main__":
    train()
