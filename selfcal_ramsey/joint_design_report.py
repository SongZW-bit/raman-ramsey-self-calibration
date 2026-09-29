"""Report matched-estimator joint-design checks and nonlinear noise audit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from two_beam_quadratic import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    current=json.loads((OUT/'two_beam_quadratic_joint_mc.json').read_text())
    old=json.loads((OUT/'two_beam_old_baseline_quadratic_mc.json').read_text())
    audit=json.loads((OUT/'two_beam_local_audit.json').read_text())
    assert len(current['results'])==10 and len(old['results'])==5
    groups=[old['results']]+[[r for r in current['results'] if r['family']==k] for k in ('hyper','shape')]
    worst=[max(g,key=lambda r:r['delta_mse']) for g in groups]
    reduction=1-worst[2]['delta_mse']/min(worst[0]['delta_mse'],worst[1]['delta_mse'])
    design_source=json.loads((OUT/'two_beam_quadratic_joint.json').read_text())
    oracle=[]
    for d in design_source['results']:
        for b in (-.6,0.,.6):
            r=statistics(d['parameters'],d['family'],d['pattern'],b,design_source['config'],3,True)
            a=r['jacobian'][:,0]
            known=1/(a@np.linalg.solve(r['covariance'],a))
            oracle.append(dict(family=d['family'],b=b,unknown_b_variance=r['variance'],
                               known_b_variance=float(known),relative_reduction=float(1-known/r['variance'])))
    summary=dict(worst_rows=worst,matched_worst_mse_reduction=reduction,
                 matched_worst_rmse_reduction=float(1-np.sqrt(1-reduction)),
                 fixed_design_known_b_diagnostic=oracle,
                 scope='Five true b points; fit range [-0.8,0.8]; all three use the same corrected Gaussian estimator.')
    (OUT/'two_beam_joint_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.8),layout='constrained')
    labels=['Previous composite','Joint composite','Joint phase-shaped']
    colors=['#7e8589','#226879','#ad4256']
    for group,label,color in zip(groups,labels,colors):
        group=sorted(group,key=lambda r:r['b'])
        axes[0].errorbar([r['b'] for r in group],[r['delta_mse']/1e-8 for r in group],
            yerr=[1.96*r['mse_standard_error']/1e-8 for r in group],fmt='o-',capsize=2,label=label,color=color)
    axes[0].set(xlabel='True light-shift parameter b',ylabel='Frequency MSE / $10^{-8}$',
                title='Matched nonlinear inference, 2000 records / point')
    axes[0].legend(frameon=False,fontsize=8)
    record=next(r for r in audit['results'] if r['family']=='hyper' and r['b']==-.6)
    vals=[record[k]/1e-8 for k in ['linear_variance','quadratic_surrogate_variance','empirical_oracle_linear_variance']]
    axes[1].bar(range(3),vals,color=['#7e8589','#75813b','#226879'],width=.6)
    axes[1].errorbar([2],[vals[2]],yerr=[1.96*record['empirical_variance_standard_error']/1e-8],
                     fmt='none',ecolor='.2',capsize=3)
    axes[1].set(xticks=range(3),xticklabels=['Linear','Quadratic','Trajectory MC'],
                ylabel='Fixed-weight variance / $10^{-8}$',
                title='Earlier candidate: curvature diagnostic at b = -0.6')
    fig.savefig(OUT/'two_beam_joint_assessment.png',dpi=200)
    fig.savefig(OUT/'two_beam_joint_assessment.pdf')
    lines=['# 联合辨识、测量排序与二阶噪声：本轮结论','',
        '本轮已完成两类脉冲同等自由度的相位/脉冲优化、固定计数下的测量排序枚举、二阶噪声代价修正，以及有限原子数的非线性估计检验。中等以上、宽条件下的优势仍未建立。','',
        '## 同等预算和相同估计器的结果','',
        '每次记录16次原子系综测量，共160000个原子；时间上限2048，光曝光上限160，名义耦合峰值上限1。实际曝光不必相等，但都满足同一上限。准备时间100/次。OU强度RMS为3%，相关时间300，两束光独立。','',
        '真实光移取 -0.6、-0.3、0、0.3、0.6，真实频率偏移0.0005；每点2000次模拟，拟合范围为[-0.8,0.8]。所有方案均采用同一二阶均值/协方差修正的非线性高斯估计器。','',
        '|方案|五个测试点的最大MSE|该点Monte Carlo标准误差|',
        '|---|---:|---:|']
    for label,row in zip(['此前较强复合脉冲','本轮联合优化复合脉冲','本轮联合优化整形脉冲'],worst):
        lines.append(f"|{label}|{row['delta_mse']:.6e}|{row['mse_standard_error']:.3e}|")
    lines.extend(['','本轮两个最终设计的实际资源：','',
                  '|方案|总原子数|总时间|光曝光|最坏名义总光强峰值|',
                  '|---|---:|---:|---:|---:|'])
    for d in design_source['results']:
        r=d['resources'];label='复合脉冲' if d['family']=='hyper' else '整形脉冲'
        lines.append(f"|{label}|{r['atoms']}|{r['elapsed']:.3f}|{r['exposure']:.4f}|{r['peak']:.4f}|")
    lines.extend(['',f'相对本轮更强的对照，最大MSE的点估计改善{reduction:.1%}，RMSE改善{1-np.sqrt(1-reduction):.1%}。这不是连续光移区间的最坏情况证明；未达到此前约25% MSE改善的内部筛选目标。','',
        '## 新增方法与发现','',
        '1. 两类脉冲均开放6类短校准相位和2类长测频相位，同时优化脉冲、暗时间和原子配额。扫描远处参数分支，并通过约束交换加入危险参数对。有限网格的马氏距离只是辨识诊断，不是全局误判概率界。','',
        '2. 对每类方案枚举12次短校准和4次长测频的全部1820种排列，再对排名靠前的候选重新优化连续控制。局部排序只用13个光移点，前24名再用81点复核，不能称连续参数的全局最优。最终估计允许使用测频后的校准，属于离线光谱；所有这些校准均计入资源。','',
        '3. 发现一阶噪声模型的优化漏洞：一阶导数下降并不意味着曲率也下降。对一个先前候选、使用相同且已知真实光移的局部权重，一阶预测方差3.9725e-8、二阶预测5.4557e-8、12000次完整轨迹模拟5.5308e-8。这个诊断解释模型差异，不是可实现估计器的性能对比。','',
        '4. 已把Hessian产生的协方差及条件二项分布方差修正加入设计和估计。独立标量/批量实现、噪声幅度缩放、权重约束、方差分解和SciPy独立拟合检查通过。理论推导见 Quadratic_noise_theory.md。','',
        '## 边界和下一步','',
        '二阶近似还不包含一阶-三阶响应的交叉协方差；设计使用无噪声均值的参数导数。实际推断仍是近似高斯似然。模型为单激发态远失谐Lambda有效模型，没有具体原子的多能级散射，也没有完整时钟伺服。当前结果不能泛化到所有文献方案。','',
        '在冻结当前脉冲和采样设置时，假想精确知道静态光移，只能再降低约1%至8%的局部方差。这不限制重新设计脉冲后的收益，但说明仅提高静态光移拟合精度不足以期待大幅提升。','',
        '下一阶段应测试能改变物理响应的强度比调制，或根据前段真实观测选择后段脉冲的两阶段策略。保留已有强对照；新策略同样计入校准成本，并同时控制梯度、曲率和错误参数分支。事件触发校准尚未实现。','',
        '## 复现入口','',
        '```text',
        'python selfcal_ramsey/quadratic_validate.py',
        'python selfcal_ramsey/joint_design_report.py',
        '```','',
        '原始结果：results/two_beam_quadratic_joint_mc.json、results/two_beam_old_baseline_quadratic_mc.json、results/two_beam_local_audit.json。'])
    (OUT.parent/'Joint_design_assessment_2026-09-18.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='fixed_design_known_b_diagnostic'},indent=2))


if __name__=='__main__':main()
