# Grouped Strategy Run Summary

- source: `experiment_logs/grouped_strategy_run_20260527-115521.csv`
- epochs: 20
- batch_size: 128
- fixed activation: ReLU
- fixed init: He

| strategy | optimizer | final val acc | best val acc | best epoch | train acc | val loss | final lr | params | BN | Dropout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `adam_baseline` | Adam | 98.50% | 98.50% | 20 | 99.88% | 0.0499 | 0.001000 | 537,354 | True | True |
| `sgd_lr_0_01` | SGD | 94.42% | 94.53% | 18 | 94.62% | 0.1812 | 0.010000 | 537,354 | True | True |
| `no_batchnorm` | Adam | 98.45% | 98.52% | 14 | 99.78% | 0.0572 | 0.001000 | 535,818 | False | True |
| `no_dropout` | Adam | 98.17% | 98.36% | 16 | 99.88% | 0.0773 | 0.001000 | 537,354 | True | False |
| `adam_lr_0_01` | Adam | 98.31% | 98.38% | 18 | 99.71% | 0.0578 | 0.010000 | 537,354 | True | True |
| `adam_lr_decay` | Adam | 98.37% | 98.41% | 12 | 99.54% | 0.0515 | 0.000001 | 537,354 | True | True |
