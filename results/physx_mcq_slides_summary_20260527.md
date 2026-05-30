# PhysX-MCQ 数据集与实验结果 Slides Summary

日期：2026-05-27  
工作目录：`/root/autodl-tmp/TTRV`

这个文件把 `results/` 下当前保留的结果压缩成 slides 可用版本，重点回答两个问题：

1. PhysX/PhysXNet 多选数据集是怎么构造的？
2. 两轮实验，也就是原始/按类型实验与 cleaned 数据实验，结果是什么？

## 1. PhysX-MCQ 数据集如何构造

### 数据来源

原始数据来自本地 PhysXNet 派生数据和 OBJ 几何资产：

```text
/root/autodl-tmp/TTRV/physx_mcq_workspace/PhysX-3D/outputs/physxnet_mcq_verl.json
/root/autodl-tmp/TTRV/physx_mcq_workspace/PhysX-3D/outputs/physxnet_mcq.jsonl
/root/autodl-tmp/TTRV/physx_mcq_workspace/PhysX-3D/dataset_toolkits/physxnet/version_1/partseg/<object_id>/objs/*.obj
```

原始 MCQ 数据规模：

| question_type | count |
| --- | ---: |
| object_category | 32,047 |
| joint_type | 9,118 |
| movable_part | 3,322 |
| total | 44,487 |

### PhysXNet 原始标签是什么

每个 PhysXNet object 的 annotation JSON 里主要有三层信息：

1. **Object-level label**
   - `object_name`: 具体物体名，例如 `Combat Knife`, `Folding Knife`, `Rapier`。
   - `category`: 更粗的类别，例如 `Tool`, `Weapon`, `Plumbing Fixture`。
2. **Part-level labels**
   - `parts`: 每个 part 有 `label` 和 `name`。
   - 例如 part label `0` 对应 `Blade`，part label `1` 对应 `Switch` 等。
3. **Kinematic group / joint labels**
   - `group_info`: 描述哪些 part 属于一个可动 group，以及这个 group 的运动类型。
   - 一个 group 里会包含 part label 列表、group id、raw kinematic type。
   - raw kinematic type 会映射到标准 joint label。

原始 kinematic label 映射如下：

| raw label | normalized label | displayed option text |
| --- | --- | --- |
| A | `no_movement_constraints` | no movement constraints |
| B | `prismatic_joint` | prismatic joint |
| C | `revolute_joint` | revolute joint |
| CB | `prismatic_and_revolute_joint` | prismatic and revolute joint |
| D | `hinge_joint` | hinge joint |
| E | `rigid_joint` | rigid joint |

这里后来出问题的核心是：一些 displayed labels 在语义上不是严格互斥的。例如 `revolute joint` 和 `hinge joint` 很接近，`rigid joint` 和 `no movement constraints` 也容易被模型视为同义或近义。

### 为什么构造成 MCQ

最开始构造 MCQ 是为了把 PhysXNet 的结构化标签转成可以用 VLM/PointLLM 直接回答的选择题，同时避免 open-ended 文本评估过于模糊。

具体目标：

- 把 object/category、part、joint 等结构化物理标签变成统一 QA 格式。
- 每题 4 个选项，答案是 `A/B/C/D`，便于自动评估。
- 保留 freeform generation 路径，因为后续 TTRL/Geo-Harmony 要从 raw response 中解析 prediction，再做 self-consistency reward。
- 选项必须来自同一标签空间，避免模型靠类型不匹配排除答案。例如 joint_type 的干扰项也必须是 joint labels，而不能混入 object names。

原始配置：

```text
num_choices=4
answer_balance=true
seed=42
shuffle_choices=true
enabled_question_types = object_category, movable_part, joint_type
```

`answer_balance=true` 表示正确答案位置按样本编号轮转，尽量均衡落在 A/B/C/D，避免 letter prior。

### 选项是如何选择的

所有题目都遵循同一个基础规则：

```text
1 correct answer + 3 distractors -> shuffle / balanced answer position -> ensure 4 unique choices
```

构造时会做基本合法性检查：

- 4 个选项必须非空。
- normalized 后不能重复。
- 正确答案必须且只能出现一次。
- distractor 不能和 answer 相同。

#### object_category 选项逻辑

`object_category` 实际使用的是 object-level 的 `object_name` 作为 answer，而不是粗粒度 `category`。

流程：

```text
answer = current_object.object_name
distractor_pool = all other object_name labels from PhysXNet
sample 3 unique distractors
```

示例：

```text
object_id=42
object_name=Combat Knife
answer=Combat Knife
distractors sampled from other object names:
  Bathroom Sink Cabinet
  Mobile Table
  Decorative Container with Mesh Cover
```

为什么这样选：

- 同属于 object name 标签空间，问题明确。
- 干扰项来自全局 object-name pool，保证有足够候选。
- 这种题主要评估整体 3D shape/category recognition。

#### movable_part 选项逻辑

`movable_part` 的 answer 来自该 object 中唯一的 known movable kinematic group。

原始构造会先筛对象：

```text
必须有 known kinematic group
默认 skip_multiple_correct=true
如果一个 object 有多个 movable groups，则跳过
如果 kinematic group 对应多个 part labels，也跳过
```

保留下来的样本满足：

```text
answer = the single part_name in the movable group
```

distractor pool 的优先级：

```text
1. same object 的其他 part names
2. 同 category 其他 object 的 part names
3. 全局 part-name pool
```

配置里有：

```text
prefer_same_object_distractors=true
min_same_object_distractors=3
```

原始构造虽然优先 same-object distractors，但如果不够，会补同类/全局 part labels。cleaned v1/v2 后更严格：只保留有足够同物体干扰项、且 part label 更可见/更可区分的样本。

为什么这样选：

- `movable_part` 的问题是“这个物体里哪个部件可动”，所以最合理的 distractors 是同一个 object 的其他 part。
- 如果干扰项来自完全不同 object，问题会变简单，模型可能靠文本先验排除。
- 同物体 distractors 更能评估模型是否理解 3D 部件和运动 group 的对应关系。

#### joint_type 选项逻辑

`joint_type` 的 answer 来自 kinematic group 的 raw label 映射。

流程：

```text
for each movable group:
  raw kinematic label -> normalized label -> displayed option text
  answer = displayed option text
  distractor_pool = all displayed kinematic labels
```

原始 distractor pool 是固定的 joint label set：

```text
no movement constraints
prismatic joint
revolute joint
prismatic and revolute joint
hinge joint
rigid joint
```

每道 joint_type 题从这个同类型 label set 中选 1 个正确 joint + 3 个 joint distractors。

为什么这样选：

- 所有选项都属于 joint/kinematic 类型，类型一致。
- 题目考察的是某个 target part 的运动约束，而不是 object category 或 part identity。
- 固定 joint label pool 可以覆盖常见运动类型，也便于统计 confusion matrix。

但这个设计后来暴露出一个问题：PhysXNet 的 raw labels 本身含有近义/重叠标签。比如同一题里可能同时出现 `revolute joint` 和 `hinge joint`，模型很容易稳定选到语义近似但非标注答案的选项。因此 cleaned v1/v2 做了 canonical 化和互斥化。

### 样本格式

每道题被组织成 4-choice MCQ：

```text
prompt: 含 <image>、题干、A/B/C/D options、只输出选项字母的指令
answer: A/B/C/D
source/id: 原始追踪字段
sidecar: object_id, question_type, part label, true answer text, debug metadata
```

注意：`image_path` 只是兼容字段。PointLLM 实际输入不是 PNG，而是从本地 OBJ 采样出来的 3D point cloud。

### 三类 MCQ 示例

下面是真实样本的简化展示。每个样本都有一个正确 letter，其余三个是 distractors。

#### object_category

`object_category` 题目询问整个物体类别。正确选项来自该 object 的类别标注，干扰项来自其他物体类别。

Example 1:

```text
Question: What is this object?

A. Combat Knife
B. Bathroom Sink Cabinet
C. Mobile Table
D. Decorative Container with Mesh Cover

Answer: A
object_id: 42
```

Example 2:

```text
Question: What type of object is shown here?

A. Corner Desk with Hutch
B. Rapier
C. Curved Reception Desk
D. Wireframe Lamp

Answer: B
object_id: 97
```

#### joint_type

`joint_type` 题目询问某个部件的运动/约束类型。原始版本直接使用 PhysXNet joint label 及同类干扰项，因此会出现语义重叠。

原始 example：

```text
Question: What kind of joint is associated with the Blade?

A. revolute joint
B. no movement constraints
C. prismatic and revolute joint
D. hinge joint

Answer: A
object_id: 103
target part: Blade
```

这个例子里 `revolute joint` 和 `hinge joint` 在语义上高度接近。模型如果回答 D，物理语义未必离谱，但 MCQ letter 会被判错。这正是 cleaned 数据要处理的问题。

cleaned v2 后同一类题会改成更互斥的 plain-language 选项：

```text
Question: What kind of joint is associated with the Blade?

A. combined sliding and rotating joint
B. rotating hinge joint
C. sliding joint (linear motion)
D. stationary or fixed joint (no relative motion)

Answer: B
object_id: 103
cleaning_reason: joint_type_canonical_mutually_exclusive
```

另一个 cleaned v2 example：

```text
Question: How is the Blade kinematically constrained?

A. rotating hinge joint
B. combined sliding and rotating joint
C. sliding joint (linear motion)
D. stationary or fixed joint (no relative motion)

Answer: A
object_id: 109
```

#### movable_part

`movable_part` 题目询问哪个 part 属于可运动/kinematic group。正确选项来自该 object 的可运动部件标注，干扰项应来自同一个物体的其他 part label。

原始 example：

```text
Question: Which part is associated with a kinematic group in this object?

A. Seat Support (Left)
B. Backrest_Bar_CenterLeft
C. back_connector
D. Blade

Answer: D
object_id: 103
```

cleaned v2 会保留有足够同物体干扰项、且 part label 更可见/更可区分的样本：

```text
Question: Which part is associated with a kinematic group in this object?

A. Vertical Support
B. Switch
C. Tube
D. Mouth

Answer: B
object_id: 165
cleaning_reason: movable_part_same_object_distractors
```

再一个 cleaned v2 example：

```text
Question: Which part is associated with a kinematic group in this object?

A. mouth
B. vertical support
C. switch
D. tube

Answer: C
object_id: 265
```

### PointLLM 输入

评估和训练都使用 PointLLM-7B：

```text
POINT_SCOPE=full_object
POINTNUM=8192
point cloud: OBJ surface sampling -> normalize XYZ to unit sphere -> append zero RGB
shape: [8192, 6]
```

主评估路径是 freeform generation：

```text
prediction_raw -> parse_prediction(prediction_raw) -> prediction in {A,B,C,D,unknown}
```

`unknown` 计为 invalid 和 incorrect。TTRL/Geo-Harmony 统计 reward 时必须基于 parsed prediction，而不是 raw response 字符串。

### 原始 MCQ 的主要问题

原始选项里有不少语义重叠或标注口径冲突，尤其是 `joint_type`：

- `hinge` vs `revolute`
- `slider` vs `prismatic`
- `rigid` vs `fixed/no movement`
- `compound prismatic+revolute` vs 单独 prismatic/revolute

这些冲突会让模型稳定输出一个语义上合理但和标注 letter 不一致的答案。TTRL/Geo-Harmony 的 self-reward 会把这种稳定错误继续 sharpen。

## 2. Cleaned 数据集版本

### Cleaned v1

路径：

```text
/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1_keep/
```

构造规则：

- 移除 `object_category`，先聚焦更物理相关的 `joint_type` 和 `movable_part`。
- `joint_type` 合并为互斥 canonical 标签。
- `movable_part` 只保留有足够同物体 distractors 的样本。
- 原始 PhysXNet 文件不改，只生成新 dataset/sidecar/parquet。

规模：

| question_type | original | cleaned v1 | skipped |
| --- | ---: | ---: | ---: |
| joint_type | 9,118 | 9,118 | 0 |
| movable_part | 3,322 | 1,454 | 1,868 |
| object_category | 32,047 | 0 | 32,047 |
| total | 44,487 | 10,572 | 33,915 |

### Cleaned v2

路径：

```text
/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/
```

当前主用版本：

```text
variant: pragmatic_plain_balanced
dataset: /root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/pragmatic_plain_balanced/
```

构造规则：

- 在 v1 基础上进一步过滤低质量样本。
- 去掉 static/no-movement joint 中标注或选项口径不稳的部分。
- 去掉 answer label 低可见、choice label 太泛、bad choice label、part label normalize 后重复的样本。
- 保留更 pragmatic、plain-language、balanced 的 `joint_type + movable_part` 子集。

规模：

| question_type | cleaned v2 pragmatic_plain_balanced |
| --- | ---: |
| joint_type | 7,669 |
| movable_part | 696 |
| total | 8,365 |

主要过滤原因：

| skipped reason | count |
| --- | ---: |
| drop_static_no_movement_joint | 1,449 |
| drop_low_visibility_answer_label | 554 |
| drop_too_many_generic_choice_labels | 167 |
| duplicate_after_part_label_normalization | 33 |
| drop_bad_choice_label | 4 |

### v1/v2 到底改了什么、删了什么

这里要区分两种操作：

```text
修改选项 = 原始样本仍可信，只是选项表述/标签集合不互斥。
删除样本 = 样本本身不适合可靠 MCQ 评估，强行改选项会引入新噪声。
```

#### v0 -> v1：主要是“修改选项”，少量按任务过滤

v1 的主要目的不是缩小 joint_type，而是把 joint_type 的标签体系 canonicalize。

| operation | affected type | count | reason |
| --- | --- | ---: | --- |
| 修改选项 | joint_type | 9,118 | raw joint labels 有近义冲突，合并为互斥 canonical labels |
| 保留但重采样选项 | movable_part | 1,454 | 只保留有同物体 distractors 的样本，使题目更像“在这个物体里选哪个 part” |
| 删除 | movable_part | 1,868 | 不足 3 个同物体干扰项，强行从全局 part pool 采样会让题目变简单/不自然 |
| 暂不纳入 | object_category | 32,047 | v1 聚焦物理相关的 joint/part，不是因为 object_category 标签不可用 |

joint_type v1 的修改例子：

```text
original choices:
A. revolute joint
B. no movement constraints
C. prismatic and revolute joint
D. hinge joint

v1 choices:
A. revolute or hinge joint
B. fixed or rigid joint (no movement)
C. prismatic joint
D. compound prismatic and revolute joint
```

这类样本不删除，因为 target part、kinematic label 都是可追溯的；问题只是 raw labels 中 `hinge/revolute`、`rigid/no movement` 等表述不互斥，所以用 canonical label 合并更合理。

#### v1 -> v2：同时“改写选项”和“删除低质量样本”

v2 的目标是得到更适合测试方法的 pragmatic subset。它不仅改写选项，还删除一些即使改选项也不可靠的样本。

v2 主版本 `pragmatic_plain_balanced` 数量变化：

| question_type | v1 count | v2 kept | v2 removed |
| --- | ---: | ---: | ---: |
| joint_type | 9,118 | 7,669 | 1,449 |
| movable_part | 1,454 | 696 | 758 |
| total | 10,572 | 8,365 | 2,207 |

v2 删除依据：

| removal reason | type | count | why delete instead of modifying |
| --- | --- | ---: | --- |
| drop_static_no_movement_joint | joint_type | 1,449 | static/no-movement 与 fixed/rigid/no constraint 口径最不稳定；如果继续保留，模型会学到标注口径而不是物理理解 |
| drop_low_visibility_answer_label | movable_part | 554 | answer part label 太难从 full-object point cloud 中可靠定位；改 distractor 不能解决视觉不可见问题 |
| drop_too_many_generic_choice_labels | movable_part | 167 | 选项如 generic support/tube/panel 等过泛，多个 part 名称语义接近；改写可能引入主观规则 |
| duplicate_after_part_label_normalization | movable_part | 33 | label normalize 后重复，无法保证 4 个唯一选项 |
| drop_bad_choice_label | movable_part | 4 | choice label 本身异常或不可用 |

所以 v2 的逻辑是：

```text
能通过 canonicalization 解决的，修改选项；
无法通过选项修改解决的，删除样本。
```

具体例子：

```text
joint_type 可以改：
  revolute joint + hinge joint -> rotating hinge joint
  rigid joint + no movement constraints -> stationary/fixed joint

movable_part 低可见样本不能简单改：
  如果目标 part 在 full-object point cloud 里不可辨认，
  即使把 distractor 改得更好，模型也没有足够视觉证据回答。
```

v2 的 cleaned joint_type 选项更偏 plain language：

```text
A. combined sliding and rotating joint
B. rotating hinge joint
C. sliding joint (linear motion)
D. stationary or fixed joint (no relative motion)
```

这比 v1 的 canonical label 更适合模型解析，也更接近互斥物理动作类别。

## 3. Round 1：原始数据 / 按 Question Type 实验

结果来源：

```text
/root/autodl-tmp/TTRV/results/20260524_physx_ttrl_majority_results_keep.md
/root/autodl-tmp/TTRV/results/physx_geo_harmony_qtype_repro_20260526/summary.md
```

### 3.1 原始全量 TTRL majority

在原始 44,487 样本上，PointLLM step0 freeform baseline 约 37.7%。TTRL majority 更新后下降。

| run | step | n | acc@1 | delta vs step0 | invalid | hit_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valall parserfix_base | 0 | 44,487 | 0.3768 | 0.0000 | 0.0167 | 0.0353 |
| valall parserfix_base | 1 | 44,487 | 0.3659 | -0.0109 | 0.0122 | 0.0348 |
| valall parserfix_base | 2 | 44,487 | 0.3433 | -0.0335 | 0.0221 | 0.0400 |
| gate075_lr5e9_kl0p2 | 0 | 44,487 | 0.3772 | 0.0000 | 0.0167 | 0.0357 |
| gate075_lr5e9_kl0p2 | 1 | 44,487 | 0.3652 | -0.0120 | 0.0123 | 0.0335 |

结论：

- TTRL majority 没有稳定涨点。
- 主要问题不是 invalid，而是 known-but-wrong。
- 模型会对同一 prompt 稳定输出错误 majority，TTRL reward 会把这个错误进一步放大。

典型坏例：

```text
physxnet_mcq_00006912
ground truth: D
step2 majority pseudo-label: A, 19/20 = 0.95
结果：错误 A 获得 reward=1.0，被继续强化。
```

### 3.2 按 question_type 的 Geo-Harmony / vote8 结果

| question_type | final strategy | fixed step0 | best acc | delta | note |
| --- | --- | ---: | ---: | ---: | --- |
| object_category | Geo-Harmony weight update, step4 | 0.395107 | 0.403002 | +0.007895 | 真正的 acc@1 权重更新涨点 |
| joint_type | test-time vote8 majority | 0.351393 | 0.365760 | +0.014367 | 不是训练涨点，是 no-update majority vote |
| movable_part | test-time vote8 majority | 0.267610 | 0.273028 | +0.005418 | 不是训练涨点，是 no-update majority vote |

关键解释：

- `object_category` 能承受小学习率 Geo-Harmony 更新，出现 acc@1 提升。
- `joint_type` 和 `movable_part` 的正向结果来自 vote8 majority，不是权重更新。
- vote8 的 `pass@8` 很高，说明采样里经常出现正确答案；但 majority 不一定正确，说明 selector 是核心问题。

### Round 1 Takeaway

```text
TTRL/Geo-Harmony 更像对模型当前输出分布做 sharpen。
如果初始分布正确，可能涨点；
如果稳定多数是错的，就会坍塌。
```

## 4. Round 2：Cleaned 数据实验

结果来源：

```text
/root/autodl-tmp/TTRV/results/20260526_cleaned_v2_geo_ttrl_comparison_keep/summary.md
```

### 4.1 Cleaned v1 静态收益

在相同 retained subset 上比较原始选项和 cleaned v1 选项：

| result | n | acc@1 | invalid | hit_max | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| original retained step0 | 10,572 | 0.335887 | 0.013999 | 0.039444 | baseline |
| cleaned v1 step0 | 10,572 | 0.397654 | 0.010878 | 0.099981 | +0.061767 |

按类型：

| question_type | original retained acc | cleaned v1 acc | delta |
| --- | ---: | ---: | ---: |
| joint_type | 0.351393 | 0.419061 | +0.067668 |
| movable_part | 0.238652 | 0.263411 | +0.024759 |

结论：

```text
数据清洗本身显著提高 static acc@1。
这说明原始 bad case 里确实有大量 option 设计/标注口径问题。
```

### 4.2 Cleaned v1 上继续跑 TTRL / Geo-Harmony

| method | step | acc@1 | joint_acc | movable_acc | delta vs step0 | early_stop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| static_step0_cache | 0 | 0.397654 | 0.419061 | 0.263411 | +0.000000 | False |
| geo_harmony | 1 | 0.389803 | 0.410178 | 0.262036 | -0.007851 | True, accuracy_drop |
| ttrl_majority | 1 | 0.386587 | 0.406339 | 0.262724 | -0.011067 | True, accuracy_drop |

结论：

- cleaned v1 改善了静态数据质量。
- 但 hard-label Geo-Harmony/TTRL 继续训练仍然掉点。
- 数据变干净后，主要矛盾转移为 pseudo-label selector 质量。

### 4.3 Cleaned v2 主 baseline

Cleaned v2 pragmatic subset 使用 fresh full static evaluation：

| dataset | n | acc@1 | invalid | hit_max |
| --- | ---: | ---: | ---: | ---: |
| cleaned v2 pragmatic balanced step0 | 8,365 | 0.500538 | 0.015780 | 0.057621 |

按类型：

| question_type | n | acc@1 | invalid | hit_max |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 7,669 | 0.522363 | 0.016299 | 0.058547 |
| movable_part | 696 | 0.260057 | 0.010057 | 0.047414 |

这是当前 cleaned 数据上的主要 step0 cache：

```text
/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/runs/20260526_v2_pragmatic_plain_balanced_static_4gpu_keep/validation_predictions_step0_reparsed_v2parser.jsonl
```

### 4.4 Cleaned v2 上的 Geo-Harmony 诊断

Loose gate：`GEO_MIN_HM=0.60`

| step | acc@1 | invalid | hit_max | joint_acc | movable_acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.500538 | 0.015780 | 0.057621 | 0.522363 | 0.260057 |
| 1 | 0.501614 | 0.014585 | 0.059175 | 0.523797 | 0.257184 |
| 2 | 0.499821 | 0.012074 | 0.059414 | 0.521320 | 0.262931 |

Strict gate：`GEO_MIN_VIEW_PROB=0.75`

| step | acc@1 | invalid | hit_max | joint_acc | movable_acc | note |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.500538 | 0.015780 | 0.057621 | 0.522363 | 0.260057 | baseline |
| 1 | 0.503288 | 0.013270 | 0.060610 | 0.525492 | 0.258621 | best observed, but zero reward/no robust training signal |
| 2 | 0.498267 | 0.011476 | 0.057262 | 0.520668 | 0.251437 | collapse / early stop |

Best observed number:

```text
strict step1 acc@1 = 0.503288
delta vs step0 = +0.002750
```

但这个不能作为稳定方法提升，因为：

- step1 训练 selector selected groups = 0，reward 基本为 0。
- step2 出现错误伪标签后，acc 掉到 0.498267。
- 这说明 hard-label Geo-Harmony gate 只能减少错标，不能根治两视角都稳定错的问题。

### 4.5 Pass@8 / Majority@8 诊断

Strict run 训练诊断：

| step | Pass@8 | Majority@8 | response_acc | selected groups | selected precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000000 | 0.500000 | 0.406250 | 0 | n/a |
| 2 | 0.500000 | 0.000000 | 0.125000 | 1 | 0.000000 |

关键解释：

- `Pass@8` 高说明正确答案确实经常出现在采样中。
- `Majority@8` 低说明多数投票不可靠。
- hard pseudo-label selector 会把少数正确答案压掉。

典型 cleaned v2 坏例：

```text
physxnet_mcq_00000023
answer: D
preds: C6, D1, unknown1
geo selected: C
geo_hm: 0.750001
结果：错误 C 被当成高置信伪标签。
```

## 5. 总结：所有 results 的统一结论

### 已确认有效

1. 数据清洗有效：

```text
v1 retained subset: 0.335887 -> 0.397654, +6.18 points
v2 pragmatic static baseline: 0.500538
```

2. 原始 object_category 上，小学习率 Geo-Harmony 权重更新有真实 acc@1 涨点：

```text
0.395107 -> 0.403002, +0.007895
```

3. 对 joint_type / movable_part，no-update vote8 majority 能涨一点：

```text
joint_type: 0.351393 -> 0.365760, +0.014367
movable_part: 0.267610 -> 0.273028, +0.005418
```

### 没有确认有效

1. 原始全量 TTRL majority 权重更新没有涨点：

```text
0.3772 -> 0.3652, -0.0120
```

2. cleaned v1 上 TTRL/Geo-Harmony 权重更新仍掉点：

```text
Geo-Harmony: 0.397654 -> 0.389803
TTRL majority: 0.397654 -> 0.386587
```

3. cleaned v2 hard-label Geo-Harmony 没有稳定涨点：

```text
best observed step1: 0.503288, but reward=0/no robust signal
step2 after wrong pseudo-label: 0.498267
```

### 核心失败机制

```text
TTRL / hard-label Geo-Harmony 本质上是在 sharpen 当前预测分布。
当模型多数采样正确时，它可能提升；
当模型多数采样稳定但错误时，它会强化错误答案。
```

当前数据中经常出现：

```text
Pass@8 高，Majority@8 低
```

这说明 PointLLM 不是完全采不到正确答案，而是伪标签选择器把 minority correct 丢掉了。

### 4.6 Pass@8 / Majority@8 汇总表

为了讲清楚“模型是否采得到正确答案”和“伪标签选择器是否选对”，需要同时看：

```text
Pass@8: 8 次采样里只要有一次正确，就算正确。它是 oracle diagnostic。
Majority@8: 8 次采样的多数票是否正确。它是可部署 vote selector 的近似。
acc@1: 单次 deterministic/freeform validation 的正确率。它是训练后主指标。
```

全量 no-update vote8 诊断：

| dataset / split | groups | response_acc | Pass@8 | Majority@8 | gap Pass-Maj | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| original joint_type vote8 | 9,118 | 0.331515 | 0.713863 | 0.365760 | +0.348103 | 经常采到正确答案，但 majority selector 很弱 |
| original movable_part vote8 | 3,322 | 0.261025 | 0.696869 | 0.273028 | +0.423841 | 正确答案常在候选里，但多数票很差 |
| cleaned v1 retained joint_type vote8 | 9,118 | n/a | 0.713863 | 0.365760 | +0.348103 | retained 原始 vote8 诊断 |
| cleaned v1 retained movable_part vote8 | 1,454 | n/a | 0.676066 | 0.249656 | +0.426410 | retained subset 中 selector 仍差 |

cleaned v2 训练 batch 诊断，注意每步只有 4 个 prompt groups，因此方差很大，不能当全量指标：

| run | train step | groups | response_acc | Pass@8 | Majority@8 | selected pseudo-label precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loose Geo-Harmony | 1 | 4 | 0.562500 | 1.000000 | 0.750000 | 1.000000 |
| loose Geo-Harmony | 2 | 4 | 0.343750 | 1.000000 | 0.000000 | 0.000000 |
| loose Geo-Harmony | 3 | 4 | 0.531250 | 1.000000 | 0.500000 | 0.666667 |
| strict Geo-Harmony | 1 | 4 | 0.406250 | 1.000000 | 0.500000 | n/a |
| strict Geo-Harmony | 2 | 4 | 0.125000 | 0.500000 | 0.000000 | 0.000000 |

关键现象：

```text
Pass@8 高，Majority@8 低：
  模型能采到正确答案，但多数票/伪标签选择器没有把正确答案选出来。

Pass@8 也低：
  模型很少采到正确答案，TTRL/Geo-Harmony 很难救。
```

为什么会出现 `Majority@8=0.75` 但 full acc@1 仍然低？

```text
1. Majority@8=0.75 来自一个训练 mini-batch，只有 4 个 prompt groups。
   也就是 3/4 个 prompt 多数票正确，样本太少，方差极大。

2. acc@1 是全量 deterministic validation。
   cleaned v2 acc@1 是 8,365 个样本上的单次输出正确率，不是同一批 4 个训练样本。

3. Majority@8 是 test-time voting 指标，不等于训练后权重更新指标。
   训练 reward 一旦选错 pseudo-label，会改变模型参数，使后续 full acc@1 下降。

4. 即使某一步 Majority@8 高，也不代表伪标签选择器持续可靠。
   loose step1 Majority@8=0.75，但 step2 变成 0.0，且 selected pseudo-label precision=0.0。
```

因此 slides 里建议这样说：

```text
The high Pass@8 shows PointLLM often samples the correct answer.
The much lower Majority@8 shows the pseudo-label selector is the bottleneck.
The occasional Majority@8=0.75 is a tiny 4-prompt training batch, not full validation.
This explains why acc@1 remains low and why hard-label updates can collapse.
```

## 6. 方法描述：从 TTRL 到 Geo-Harmony

### 基础 TTRL majority

TTRL 的基本想法是：test time 不使用 ground truth，而是对同一个 prompt 多次采样，用模型自身的 self-consistency 生成伪标签。

在 MCQ 任务中，我们不能直接比较 raw response 字符串，因为模型可能输出完整句子。因此先做解析：

```text
prediction_raw -> parse_prediction -> A/B/C/D/unknown
```

最初的 hard-label TTRL reward：

```text
sample K responses for same prompt
parse each response to A/B/C/D/unknown
majority_label = most frequent parsed answer
reward_i = 1 if prediction_i == majority_label else 0
```

伪代码：

```text
Algorithm 1: Hard-label TTRL for PhysX-MCQ

Input:
  model pi_theta
  prompt x
  point cloud p
  K rollouts

responses = []
for k = 1..K:
    y_k = pi_theta.generate(x, p, do_sample=True)
    a_k = parse_prediction(y_k)  # A/B/C/D/unknown
    responses.append(a_k)

majority_label = majority_vote(responses excluding unknown)

for k = 1..K:
    if a_k == unknown:
        reward_k = unknown_reward
    else if a_k == majority_label:
        reward_k = 1
    else:
        reward_k = 0

update pi_theta with GRPO/PPO using reward_k
```

问题：

```text
majority != correctness
```

如果模型 8 次采样里 6 次都错，hard-label reward 会把错误答案当成正标签。

### Geo-Harmony 的改进

Geo-Harmony 是我们为 3D MCQ 改造的 Self-Harmony/TTRL 思路。核心变化是：不只看同一个点云视角下的 majority，而是构造多个 point-cloud reframe views，要求伪标签在几何扰动下也一致。

当前实现使用：

```text
POINT_REFRAME_POLICY=sensor_noise
GEO_NUM_VIEWS=2
GEO_SAMPLES_PER_VIEW=4
ROLLOUT_N=8
```

也就是每个 prompt 采 8 次，其中 4 次来自 view 0，4 次来自 view 1。每个 view 是同一 OBJ point cloud 的轻微扰动版本。

Geo-Harmony hard-label score：

```text
per_view_prob_v(a) = count of answer a in view v / samples_per_view
geo_score(a) = harmonic_mean_v(per_view_prob_v(a))
geo_label = argmax_a geo_score(a)
```

只有当：

```text
geo_score(geo_label) >= GEO_MIN_HM
view support 足够
可选：每个 view 中该 label 的 prob >= GEO_MIN_VIEW_PROB
```

才给这个 label reward。

伪代码：

```text
Algorithm 2: Hard-label Geo-Harmony for PhysX-MCQ

Input:
  model pi_theta
  prompt x
  original point cloud p
  V views
  M samples per view

all_predictions = []
for v = 1..V:
    p_v = reframe_point_cloud(p, policy="sensor_noise")
    for m = 1..M:
        y_vm = pi_theta.generate(x, p_v, do_sample=True)
        a_vm = parse_prediction(y_vm)
        all_predictions.append((v, a_vm))

for each answer a in {A,B,C,D}:
    probs = []
    for v = 1..V:
        prob_v = count(a in view v) / M
        probs.append(prob_v)
    geo_score[a] = harmonic_mean(probs)

geo_label = argmax_a geo_score[a]

if geo_score[geo_label] < GEO_MIN_HM:
    skip update for this prompt
if min_view_prob is enabled and min_v prob_v(geo_label) < GEO_MIN_VIEW_PROB:
    skip update for this prompt

for each rollout prediction a_vm:
    reward_vm = 1 if a_vm == geo_label else 0

update pi_theta with GRPO/PPO using reward_vm
```

相对 TTRL majority，Geo-Harmony 的改进点：

- TTRL 只看 overall majority。
- Geo-Harmony 看不同 geometric views 下是否一致。
- 用 harmonic mean 惩罚“只在一个 view 很高、另一个 view 不支持”的标签。
- 加入 `GEO_MIN_VIEW_PROB` 后，可以过滤一部分弱一致伪标签。

### 当前 Geo-Harmony 的局限

Geo-Harmony 仍然是 hard-label reward。它能过滤一部分不稳定 pseudo-label，但挡不住“两种 view 都稳定错”的情况。

典型 cleaned v2 bad case：

```text
answer: D
sampled predictions: C6, D1, unknown1
geo selected: C
geo_hm: 0.750001
result: wrong C is reinforced
```

因此当前结果是：

```text
cleaned v2 strict step1: 0.500538 -> 0.503288
but step2: 0.498267 after one wrong pseudo-label update
```

这个 small gain 不是稳定方法提升。

### 下一步：Soft-label Geo-Harmony / RESTRAIN-lite

根据当前 failure mode，更合理的下一步是把 hard majority pseudo-label 改成 soft vote distribution。

直觉：

```text
不要把全部 reward 质量压到 majority answer。
按 vote count 给 A/B/C/D 分配 soft pseudo-label mass。
这样 minority correct 不会被完全压掉。
```

候选 soft reward：

```text
q(a) = count(a)^gamma / sum_b count(b)^gamma
reward_i = q(prediction_i)
```

结合 Geo-Harmony：

```text
geo_score(a) = harmonic_mean_v(per_view_prob_v(a))
q_geo(a) = normalize(geo_score(a)^gamma)
reward_i = q_geo(prediction_i)
```

伪代码：

```text
Algorithm 3: Soft-label Geo-Harmony / RESTRAIN-lite

Input:
  model pi_theta
  prompt x
  point cloud p
  V views, M samples per view
  gamma > 1

Collect parsed predictions a_vm exactly as in Geo-Harmony.

for each answer a in {A,B,C,D}:
    for each view v:
        prob_v[a] = count(a in view v) / M
    geo_score[a] = harmonic_mean_v(prob_v[a])

if max_a geo_score[a] < threshold:
    skip update for this prompt

q_geo[a] = geo_score[a]^gamma / sum_b geo_score[b]^gamma

for each rollout prediction a_vm:
    if a_vm == unknown:
        reward_vm = unknown_reward
    else:
        reward_vm = q_geo[a_vm]

update pi_theta with GRPO/PPO using reward_vm
```

预期收益：

- hard label: `C6/D1/unknown1` 会给 C=1, D=0。
- soft label: C 仍然更高，但 D 不会被完全归零。
- 如果 Pass@K 高但 Majority@K 低，soft reward 可能比 hard reward 更稳。

风险：

- 如果错误多数非常强，soft label 仍会主要强化错误答案，只是更缓。
- 因此还需要 prompt-level gate、early stop、acc@1 validation。

### 方法结果一句话

```text
Data cleaning gives a clear static gain. Hard-label Geo-Harmony gives only a small transient gain and then collapses on wrong pseudo-labels. The next method improvement should replace hard majority reward with a soft vote-distribution reward.
```

## 7. Slides 推荐表述

### 一页讲数据集

```text
We build PhysX-MCQ from PhysXNet by converting object/category, joint type, and movable part annotations into 4-choice MCQ questions. PointLLM sees sampled 3D OBJ point clouds, not PNG images. Each free-form answer is parsed into A/B/C/D/unknown for evaluation and test-time RL.
```

### 一页讲为什么清洗

```text
Original MCQ options contain semantic overlaps such as hinge/revolute and slider/prismatic. These make self-consistency unreliable: the model can be confidently consistent on a semantically plausible but label-mismatched option.
```

### 一页讲 clean 数据收益

```text
Cleaning options and filtering bad movable-part distractors substantially improves static accuracy: 33.6% -> 39.8% on the retained v1 subset. A stricter v2 pragmatic subset reaches 50.1% static acc@1.
```

### 一页讲方法结果

```text
Geo-Harmony improves object_category on the original data (+0.79 acc points), but hard-label TTRL/Geo-Harmony does not stably improve joint_type/movable_part after cleaning. The main bottleneck is pseudo-label selection, not simply sampling correct answers.
```

### 一页讲下一步

```text
Move from hard majority pseudo-labels to soft vote-distribution rewards. Use vote-count-weighted pseudo-label distributions and prompt-level gates so minority correct samples are not fully suppressed by a spurious majority.
```

## 8. 关键路径索引

Slides summary 本文件：

```text
/root/autodl-tmp/TTRV/results/physx_mcq_slides_summary_20260527.md
```

结果目录：

```text
/root/autodl-tmp/TTRV/results/20260524_physx_ttrl_majority_results_keep.md
/root/autodl-tmp/TTRV/results/physx_geo_harmony_qtype_repro_20260526/summary.md
/root/autodl-tmp/TTRV/results/20260526_cleaned_v2_geo_ttrl_comparison_keep/summary.md
```

Cleaned 数据：

```text
/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v1_keep/
/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/
```

当前 cleaned v2 step0 cache：

```text
/root/autodl-tmp/TTRV/outputs/physx_cleaned_options_v2_candidates_keep/runs/20260526_v2_pragmatic_plain_balanced_static_4gpu_keep/validation_predictions_step0_reparsed_v2parser.jsonl
```
