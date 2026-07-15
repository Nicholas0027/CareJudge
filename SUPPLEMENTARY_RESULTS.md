# CARE-Judge 补充实验

## 1. Qwen 空verdict分析
- Qwen 在所有 benchmarks 上空输出率 < 3%,说明模型格式正常
- Safe abstention 不是因为输出格式问题,而是真的无法区分对错

## 2. Rubric 语义一致性
- GPT-5.5: rubric_flip 仅 1-8%,说明 GPT 产出非常稳定
- DeepSeek: rubric_flip 5-20%,说明中等能力评判者更易受 rubrics 影响
- 信号方向和稳定性符合预期

## 3. δ 敏感性
- δ ∈ {0.05,0.10,0.20} 时覆盖率稳定
- 校准阈值选择对 δ 不敏感

## 4. 随机种子稳健性
- 20 seeds AUROC: 0.675±0.016 (变异系数 2.4%)
- 极小方差,实验稳定

## 5. 校准曲线
- GPT-5.5: 所有 benchmark ECE < 0.06 极好
- DeepSeek: ECE 0.16-0.29 中等
- 与预期一致: 强评判者校准好

## 6. 特征消融
- 最重要的特征: swap_consistency (去掉后 Δ=-0.061)
- 其次是: self_vote_share (Δ=-0.022), rubric_vote_share (Δ=-0.019)
- 三个信号之间有互补性,全量融合效果最好

## 7. 校准集大小
- n_cal 从 200-1000,AUROC 保持 0.676 不变
- 低数据量也能稳定工作
