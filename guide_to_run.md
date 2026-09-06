# Running the matrix on Grid5000

## 1. Launch

```
wandb sweep test/sweep/matrix.yaml       # prints SWEEP_ID
SWEEP=<the id it printed>

g5k agent $SWEEP --site grenoble --n-agents 8 \
    --walltime 3:00:00 --no-besteffort
```
## 2. Pull and render

```
g5k sync model results/$SWEEP --site grenoble

python test/plot_matrix_report.py      --run-id $SWEEP
python test/plot_divergence_report.py  --run-id $SWEEP --variant acc_basic
python test/plot_global_1deg_report.py --run-id $SWEEP     # if global_1deg ran
```

## Sizing, measured on GPU

| | mini | veros | per variant |
|---|---|---|---|
| `acc`, 10950 steps | ~9 ms/step | ~50 ms/step | ~11 min |
| `global`, 10950 steps | ~39 ms/step | ~51 ms/step | ~16 min |
| `global_1deg`, 300 steps | 224 ms/step | 397 ms/step | ~3 min |
| `global_1deg`, 1 model year | 172 ms/step | 354 ms/step | ~5.1 h |