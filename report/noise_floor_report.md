# real veros: run-to-run noise floor

Same variant (setups_matrix.py) run twice as two independent processes -- fresh interpreter each time, rules out shared JIT-cache/state artifacts -- real veros only, no mini_veros. Diffed with the same error-evolution machinery test_matrix.py uses for mini vs real.

Measured across 3 variant(s): max abs diff (run A vs run B, any field/step) = **0.00e+00**.

For contrast, test_matrix.py gates mini vs real at rel error < 1e-06. If the noise floor above is ~0 (float64 eps), that gate is not run-to-run solver noise headroom -- it's an assumed margin. Any nonzero mini/real diff above the noise floor is real divergence, not noise.

| variant | steps | max abs diff (run A vs run B, any field/step) |
|---|---|---|
| acc_basic | 6 | 0.00e+00 |
| acc_surface_pressure | 300 | 0.00e+00 |
| global_default | 60 | 0.00e+00 |

## per-field detail

### acc_basic (6 steps)

| field | max abs diff |
|---|---|
| psi | 0.00e+00 |
| salt | 0.00e+00 |
| temp | 0.00e+00 |
| tke | 0.00e+00 |
| u | 0.00e+00 |
| v | 0.00e+00 |

### acc_surface_pressure (300 steps)

| field | max abs diff |
|---|---|
| psi | 0.00e+00 |
| salt | 0.00e+00 |
| temp | 0.00e+00 |
| tke | 0.00e+00 |
| u | 0.00e+00 |
| v | 0.00e+00 |

### global_default (60 steps)

| field | max abs diff |
|---|---|
| eke | 0.00e+00 |
| psi | 0.00e+00 |
| salt | 0.00e+00 |
| temp | 0.00e+00 |
| tke | 0.00e+00 |
| u | 0.00e+00 |
| v | 0.00e+00 |

