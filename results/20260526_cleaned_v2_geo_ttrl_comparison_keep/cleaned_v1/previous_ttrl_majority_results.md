# PhysX-MCQ TTRL Majority Results - 2026-05-24

工作目录：`/root/autodl-tmp/TTRV`

## 结论

当前 PointLLM-7B 在 PhysX-MCQ freeform 上的 step0 能力不足，valall baseline 只有约 `37.7%`。TTRL majority reward 会强化同一 prompt 下重复出现的答案；当模型重复的是错误答案时，训练会快速把错误 majority 放大，导致整体 accuracy 下降。现有实验没有得到相对 step0 的稳定正向提升。

因此下一步不建议继续盲扫当前 majority reward 参数。更合理的方向是先提升/过滤伪标签质量，例如只训练高置信且经额外校验的样本，或者引入更强的 base model / reranker / choice-logits 诊断，而不是直接用 freeform majority 当伪标签。

## 核心完整结果

### Local train20/val200, parserfix_base

路径：

```text
/root/autodl-tmp/TTRV/outputs/20260524_082145_grid01_0_parserfix_base
```

参数要点：`REWARD_VARIANT=ttrl_majority_vote`，`TTRL_MIN_MAJORITY_RATIO=0.0`，`ACTOR_LR=1e-8`，`KL_LOSS_COEF=0.1`，`MAX_RESPONSE_LENGTH=24`。

| step | n | acc | delta vs step0 | invalid | hit-max | joint_type | movable_part | object_category |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 200 | 0.345 | +0.000 | 0.015 | 0.045 | 0.358 | 0.269 | 0.409 |
| 1 | 200 | 0.340 | -0.005 | 0.020 | 0.045 | 0.358 | 0.269 | 0.394 |
| 2 | 200 | 0.375 | +0.030 | 0.015 | 0.035 | 0.358 | 0.358 | 0.409 |

这个 `+3` 点只在 val200 上成立，后续 valall 没有复现，因此不能作为最终效果。

### Valall, parserfix_base

路径：

```text
/root/autodl-tmp/TTRV/outputs/20260524_202037_physx_ttrl_valall_grid_detached/parserfix_base_valall_s3_bs64
```

参数要点同 local parserfix_base，验证 batch 改为 `VAL_BATCH_SIZE=64` 只为加速。该 run 在用户要求下于 step3 partial 后停止，完整可用指标为 step0-step2。

| step | n | acc | delta vs step0 | invalid | hit-max | joint_type | movable_part | object_category |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 44,487 | 0.3768 | +0.0000 | 0.0167 | 0.0353 | 0.3514 | 0.2694 | 0.3952 |
| 1 | 44,487 | 0.3659 | -0.0109 | 0.0122 | 0.0348 | 0.3473 | 0.2631 | 0.3818 |
| 2 | 44,487 | 0.3433 | -0.0335 | 0.0221 | 0.0400 | 0.3335 | 0.2381 | 0.3570 |

step3 只写了 `1,856/44,487` 条，被人工停止，不作为正式指标。partial step3 acc 约 `0.2408`，已经呈现继续坍塌趋势。

训练 bad case 说明问题很直接：`physxnet_mcq_00006912` 的正确答案是 `D`，但 step2 中 `A` 以 `19/20 = 0.95` 的比例成为 pseudo label，错误回答 `A. Arm Near Vertical Bar (Right Front)` 获得 reward `1.0`。这类高置信错误 majority 会被继续放大。

### Valall, gate075_lr5e9_kl0p2

路径：

```text
/root/autodl-tmp/TTRV/outputs/20260524_220158_physx_ttrl_valall_grid_detached/gate075_lr5e9_kl0p2_max24_valall_s3_bs64_stop2pt
```

参数要点：`TTRL_MIN_MAJORITY_RATIO=0.75`，`ACTOR_LR=5e-9`，`KL_LOSS_COEF=0.2`，`EARLY_STOP_ACC_DROP=0.02`，`MAX_RESPONSE_LENGTH=24`。该 run 在 step2 partial 后按用户要求停止，完整可用指标为 step0-step1。

| step | n | acc | delta vs step0 | invalid | hit-max | joint_type | movable_part | object_category |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 44,487 | 0.3772 | +0.0000 | 0.0167 | 0.0357 | 0.3510 | 0.2739 | 0.3954 |
| 1 | 44,487 | 0.3652 | -0.0120 | 0.0123 | 0.0335 | 0.3405 | 0.2613 | 0.3829 |

step2 只写了 `11,136/44,487` 条，不作为正式指标。partial step2 acc 约 `0.3174`，仍然继续下降。gating 在 step1 把训练 reward 压到更保守：step1 所有训练样本 `ttrl_pseudo_label=none`，reward mean 为 `0.0`；但一次更新后 step1 valall 仍下降约 `1.2` 点，说明当前训练/模型状态本身已经很容易被扰动。

到 step2，gating 仍不能避免错误高置信 majority：同一个 `physxnet_mcq_00006912` 仍出现 `A` 作为 `0.95` majority pseudo label，而 ground truth 是 `D`。

## 主要坏例和机制

1. 模型 baseline 能力偏低，freeform 输出里经常给出完整选项文本或解释，而不是严格只输出 letter。
2. 解析器已能较好地从 raw response 中恢复 A/B/C/D；当前主要错误不是 unknown，而是 known-but-wrong。
3. Majority reward 不知道 ground truth，因此稳定错误答案会被当成好伪标签。
4. `movable_part` 和 `object_category` 最容易掉，`joint_type` 有时相对稳定但也没有可靠提升。
5. invalid/hit-max 不是这轮主要矛盾：parserfix_base step2 的 invalid 从 1.67% 到 2.21%，hit-max 从 3.53% 到 4.00%，上涨不大，但 acc 已经从 37.68% 掉到 34.33%。

## 已停止的实验

当前没有训练进程继续运行，GPU 已释放。最后两个 valall run 都是人工停止：

```text
parserfix_base_valall_s3_bs64: step3 partial 1,856/44,487
gate075_lr5e9_kl0p2_max24_valall_s3_bs64_stop2pt: step2 partial 11,136/44,487
```

这些 partial 文件只用于排查趋势，不用于正式报告。

## Outputs 清理

`/root/autodl-tmp/TTRV/outputs` 已按用户要求清理：只保留本文正式引用的实验文件夹，其他 top-level 输出目录已删除。当前保留：

```text
/root/autodl-tmp/TTRV/outputs/20260524_082145_grid01_0_parserfix_base
/root/autodl-tmp/TTRV/outputs/20260524_202037_physx_ttrl_valall_grid_detached
/root/autodl-tmp/TTRV/outputs/20260524_220158_physx_ttrl_valall_grid_detached
```

其中 valall 正式结果子目录为：

```text
/root/autodl-tmp/TTRV/outputs/20260524_202037_physx_ttrl_valall_grid_detached/parserfix_base_valall_s3_bs64
/root/autodl-tmp/TTRV/outputs/20260524_220158_physx_ttrl_valall_grid_detached/gate075_lr5e9_kl0p2_max24_valall_s3_bs64_stop2pt
```

所有 `run.log` / `*.driver.log` / `*.out` / `*.err` / `*.trace` 已从 `outputs` 中清除；之前的日志归档目录也已删除。保留实验目录中的 prediction JSONL、metrics、reports 和 run configs 仍在。

## 下一步建议

优先不要再用当前设置直接网格搜索。建议按下面顺序推进：

1. 先做 base model 能力诊断：choice_logits 上限、freeform prompt 更严格化、按 question_type 分析选项文本混淆。
2. 对 majority reward 增加伪标签质量过滤：至少过滤掉 answer distribution 不稳定、选项语义相似、模型输出解释和选项文本冲突的 group。
3. 只在训练集上对 `majority_correct@votes` 较高的类型做 TTRL，例如先跳过 `movable_part`。
4. 如果继续 TTRL，先在 val200 做重复种子验证，要求 `best - step0 >= 0.02` 且不出现 step2/step3 单调下降，再跑 valall。
5. 如果要从 prompt 入手，目标不是减少 invalid，而是减少“看起来很自信的错误 majority”；应重点检查 raw prediction 与选项文本匹配错误，而不是只要求输出单字母。
