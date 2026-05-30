# PhysX Cleaned Options V2 Candidate Summary

All variants keep the VERL/sidecar schema and write new files only under this output directory.
Retained-id metrics are computed with v1 step0 predictions and original v1 answer letters; rephrased/reordered prompts require a fresh static eval.

| variant | kept | joint | movable | retained acc | invalid | hit_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pragmatic_plain_balanced | 8365 | 7669 | 696 | 0.4709 | 0.0096 | 0.0983 |
| joint_plain_no_static | 7669 | 7669 | 0 | 0.4868 | 0.0085 | 0.1031 |
| joint_motion_reliable | 5982 | 5982 | 0 | 0.5980 | 0.0094 | 0.0824 |
| movable_visible_strict | 696 | 0 | 696 | 0.2960 | 0.0216 | 0.0445 |
