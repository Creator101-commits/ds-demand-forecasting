# M5 Demand Forecasting Report

- git SHA: `2a0a33787d77f0a8437c90994bf719a5cfcfd7ed`
- Protocol: m5-benchmark-v2
- ordering_ok: True

## Validation WRMSSE (lower is better)

| Model | WRMSSE |
| :--- | ---: |
| naive | 1.4639 |
| seasonal_naive | 0.8697 |
| ma28 | 1.0823 |
| lightgbm_63x0.08 | 0.7184 |
| lightgbm_63x0.03 | 0.7152 |
| lightgbm_255x0.08 | 0.7406 |
| lightgbm_255x0.03 | 0.7477 |
| lightgbm | 0.7152 |

## Test WRMSSE

| Model | WRMSSE | 95% CI | MAE | Bias |
| :--- | ---: | :--- | ---: | ---: |
| naive | 1.7520 | [1.6267, 1.8540] | 1.47 | 0.34 |
| seasonal_naive | 0.8470 | [0.8225, 0.9035] | 1.32 | -0.02 |
| ma28 | 0.9462 | [0.9147, 0.9795] | 1.12 | 0.00 |
| lightgbm | 0.6211 | [0.5958, 0.6944] | 1.10 | 0.04 |
