# 2D TTRV Soft Pseudo-Label Idea

原本 2D TTRV 的完整实现已经在 `verl/` 里，下一步不要大改框架，只在现有 reward / pseudo-label 逻辑里替换思想。

核心想法：

1. 原 hard-label TTRV 是把全部 reward 压到 majority answer。
2. 这更像对模型原始采样分布做 sharpen，如果 majority 错，就容易继续坍塌。
3. 可以保留现有采样、rollout、PPO/GRPO 训练流程，只把 hard majority reward 改成 soft pseudo-label reward。
4. 如果需要更稳，可以用简单 2D transform 采样来判断伪标签是否可靠，但不要重写整套 pipeline。

## Simple Soft Reward

```python
# Existing 2D TTRV:
# sample K responses for prompt/image
# parse each response to A/B/C/D/unknown

preds = [parse_answer(raw) for raw in sampled_responses]

counts = {
    "A": count(preds == "A"),
    "B": count(preds == "B"),
    "C": count(preds == "C"),
    "D": count(preds == "D"),
}

# Hard label old version:
# pseudo = argmax(counts)
# reward_i = 1 if pred_i == pseudo else 0

# Soft label new version:
gamma = 2.0
denom = sum(counts[a] ** gamma for a in ["A", "B", "C", "D"])

if denom == 0:
    skip_update()

q = {
    a: (counts[a] ** gamma) / denom
    for a in ["A", "B", "C", "D"]
}

for pred_i in preds:
    if pred_i == "unknown":
        reward_i = unknown_reward
    else:
        reward_i = q[pred_i]
```

## Optional Transform Reliability

如果想利用 transform，不需要复杂实现，只是在采样时把同一个样本扩展成几个轻微 view：

```python
views = [
    original(image),
    resize_crop(image),
    mild_color_jitter(image),
    mild_blur(image),
]

all_preds = []

for view in views:
    responses = sample_model(prompt, view, n=M)
    all_preds.extend(parse_answer(r) for r in responses)

counts = count_ABCD(all_preds)

if max(counts.values()) / sum(counts.values()) < threshold:
    skip_update()
else:
    q = soft_distribution(counts, gamma=2.0)
    reward_i = q[pred_i]
```

第一轮实验只改 reward：

```text
hard majority reward -> soft vote-count reward
```

其它参数先不动。
