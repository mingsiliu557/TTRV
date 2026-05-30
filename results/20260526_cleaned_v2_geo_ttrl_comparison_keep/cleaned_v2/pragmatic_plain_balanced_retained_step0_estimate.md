# pragmatic_plain_balanced retained step0 estimate

This uses v1 step0 predictions and original v1 answer letters on retained ids. It estimates subset quality only; rephrased/reordered prompts need a fresh static evaluation.

- kept: 8365
- skipped: 2207
- retained-id acc: 0.4709
- invalid: 0.0096
- hit_max: 0.0983

## By Question Type

| question_type | n | acc | invalid | hit_max |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 7669 | 0.4868 | 0.0085 | 0.1031 |
| movable_part | 696 | 0.2960 | 0.0216 | 0.0445 |

## Skipped Reasons

- drop_bad_choice_label: 4
- drop_low_visibility_answer_label: 554
- drop_static_no_movement_joint: 1449
- drop_too_many_generic_choice_labels: 167
- duplicate_after_part_label_normalization: 33
