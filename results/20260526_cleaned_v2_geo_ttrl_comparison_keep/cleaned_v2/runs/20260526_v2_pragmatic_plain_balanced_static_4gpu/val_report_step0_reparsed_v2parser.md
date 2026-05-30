# V2 Static Reparsed Metrics

Same raw model outputs as validation_predictions_step0.jsonl, rescored after parser parenthetical-phrase fix.

- overall acc: 50.05%
- invalid: 1.58% (132/8365)
- hit_max: 5.76%
- changed by reparse: 547

| question_type | n | acc | invalid | hit_max |
| --- | ---: | ---: | ---: | ---: |
| joint_type | 7669 | 52.24% | 1.63% | 5.85% |
| movable_part | 696 | 26.01% | 1.01% | 4.74% |

## Reparse Transitions

| transition | count |
| --- | ---: |
| unknown->A | 329 |
| unknown->B | 111 |
| unknown->C | 82 |
| unknown->D | 21 |
| B->C | 2 |
| A->D | 1 |
| A->B | 1 |
