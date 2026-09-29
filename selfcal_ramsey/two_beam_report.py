"""Summarize completed finite-data checks without promoting local gains."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'results'


def main():
    final=json.loads((OUT/'two_beam_corrected_final.json').read_text())
    assert len(final['results'])==18
    wide=json.loads((OUT/'two_beam_corrected_wide.json').read_text())
    diagnostic=json.loads((OUT/'two_beam_discrimination_mc.json').read_text())
    assert len(diagnostic['results'])==6
    summary={}
    for kind in ('hyper','shape'):
        rows=[r for r in final['results'] if r['family']==kind]
        summary[kind]=max(rows,key=lambda r:r['delta_mse'])
    summary['bounded_worst_mse_reduction']=1-summary['shape']['delta_mse']/summary['hyper']['delta_mse']
    summary['bounded_worst_rmse_reduction']=1-np.sqrt(summary['shape']['delta_mse']/summary['hyper']['delta_mse'])
    summary['wide_capture_original']={r['family']:r for r in wide['results']}
    summary['diagnostic_phase_coding']=diagnostic['results']
    summary['limitations']=['Nine test b values are not a continuous worst-case guarantee.',
        'Composite pulse family and tested schedules, not all published clock servos.',
        'The Gaussian covariance and second-order mean correction remain approximate.',
        'Diagnostic phase coding is a pilot with 1000 records per condition.',
        'No complete novelty search; no global optimization certificate.']
    (OUT/'two_beam_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#23677b','#ad4256','#75813b']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for i,kind in enumerate(('hyper','shape')):
        rows=sorted((r for r in final['results'] if r['family']==kind),key=lambda r:r['b'])
        axes[0].errorbar([r['b'] for r in rows],[r['delta_mse']/1e-8 for r in rows],
            yerr=[1.96*r['mse_standard_error']/1e-8 for r in rows],fmt='o-',color=colors[i],capsize=2,
            label=['Composite baseline','Phase-shaped candidate'][i])
    axes[0].set(xlabel='Unknown light-shift parameter b',ylabel='Frequency MSE / $10^{-8}$',
                title='Known parameter range: [-0.6, 0.6]')
    axes[0].legend(frameon=False,fontsize=8)
    rows=[summary['wide_capture_original'][kind] for kind in ('hyper','shape')]
    rows.append(next(r for r in diagnostic['results'] if r['family']=='shape' and r['b']==-.6))
    axes[1].bar(np.arange(3),[r['delta_mse']/1e-8 for r in rows],color=colors,width=.6)
    axes[1].errorbar(np.arange(3),[r['delta_mse']/1e-8 for r in rows],
        yerr=[1.96*r['mse_standard_error']/1e-8 for r in rows],fmt='none',ecolor='.2',capsize=3)
    axes[1].set(xticks=np.arange(3),xticklabels=['Composite','Shaped','Shaped +\nphase coding'],
                ylabel='Frequency MSE / $10^{-8}$',title='Wider fit range: [-0.8, 0.8], b = -0.6')
    fig.savefig(OUT/'two_beam_finite_data.png',dpi=200)
    fig.savefig(OUT/'two_beam_finite_data.pdf')
    original=summary['wide_capture_original']['shape']
    baseline=summary['wide_capture_original']['hyper']
    coded=rows[-1]
    report=f'''# 两束 Raman 光独立漂移：本轮结果与下一轮主线

本轮已完成物理模型校验、原子配额优化、分组相位读出对照、非线性参数估计及二阶噪声均值修正。尚不能宣称获得了相对所有已有方案的显著优势，也没有完成具体方案的新颖性检索。

## 有限数据结果

设计范围和拟合范围均为 b∈[-0.6,0.6]；9 个光移点，每点每种方案 5000 次模拟，真实归一化频率偏移 0.0005。每次记录包含 16 次原子系综测量，总原子数严格为 160000，时间上限 2048，名义光曝光上限 160。强度 RMS 为 3%，两束光独立，OU 相关时间为 300。

|方案|9 个测试点中的最大 MSE|该点的 Monte Carlo 标准误差|
|---|---:|---:|
|已优化复合脉冲及分组读出|{summary['hyper']['delta_mse']:.6e}|{summary['hyper']['mse_standard_error']:.3e}|
|相位整形候选|{summary['shape']['delta_mse']:.6e}|{summary['shape']['mse_standard_error']:.3e}|

最大 MSE 的点估计降低 {summary['bounded_worst_mse_reduction']:.1%}，对应 RMSE 降低 {summary['bounded_worst_rmse_reduction']:.1%}。这是有限测试点上的比较，不是整个连续参数范围的最坏情况证明，也不表示每个点都改善相同比例。

## 从其他领域借鉴的第一项试验

将拟合搜索范围放宽到 [-0.8,0.8] 后，原整形方案出现远处分支误判。借鉴实验可辨识性设计，仅修改 4 次既有短校准的读出相位，专门增强参数区分能力。

在 b=-0.6，每种方案 1000 次模拟的诊断中：

|方案|MSE|光移估计误差大于 0.15 的样本比例|
|---|---:|---:|
|复合脉冲对照|{baseline['delta_mse']:.6e}|{baseline['nuisance_catastrophic_fraction']:.1%}|
|原整形方案|{original['delta_mse']:.6e}|{original['nuisance_catastrophic_fraction']:.1%}|
|整形加辨识相位编码|{coded['delta_mse']:.6e}|{coded['nuisance_catastrophic_fraction']:.1%}|

相位编码相对原整形方案修复了明显失效，但相对强对照的 MSE 点估计只降低 {1-coded['delta_mse']/baseline['delta_mse']:.1%}。不能把前者的大改善写成相对前人成果的提升。零次观察到误判不等于真实误判概率为零。该试验是固定设置，还没有闭环自适应。

## 机制与边界

相位整形候选的局部收益主要来自共模强度噪声抑制；差模噪声并未同步改善。固定脉冲移到更慢或更快漂移区间后，优势可能消失。相关时间扫描仅比较冻结的控制参数，未在每个新噪声条件重新优化。

理论上，过去校准不能预测之后新增的 OU 随机量。在静态噪声响应积分为零时，慢漂移残差的领先项为 (2S/τ)∫F(t)²dt，其中 F 为累积响应。这统一了脉冲响应与测量间隔的作用，详见 Innovation_limit.md。这是已有随机过程理论的应用，尚不构成独立新颖性。

模型仍为远失谐、单激发态 Lambda 的有效模型。三能级直接传播支持该近似的极限，但未加入具体 Rb/Cs 多能级、散射、加载时间随原子数变化或完整时钟伺服。拟合使用近似高斯协方差和二阶均值修正，并非精确的 OU 路径边缘似然。优化无全局最优证明。

## 下一轮

优先推进“主动辨识测量＋事件触发校准＋脉冲共同设计”，以约 25% 或更多最坏情况 MSE 改善作为内部筛选目标，同时推导误分支风险和有限记忆误差。光强比调制作为独立物理自由度另做消融。具体方法、成本、公平比较与已核对的文献入口见 Next_research_route.md。

## 结果文件

- results/two_beam_summary.json：可复核数值汇总。
- results/two_beam_finite_data.pdf：有限数据与误分支比较图。
- results/two_beam_mechanism.pdf：累积响应及固定控制的相关时间扫描。
- results/two_beam_validation.json、inference_corrected_validation.json、mean_correction_validation.json、innovation_validation.json：物理、数值和理论一致性检查。
'''
    (OUT.parent/'Two_beam_assessment_2026-09-18.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('diagnostic_phase_coding','wide_capture_original')},indent=2))


if __name__=='__main__':main()
