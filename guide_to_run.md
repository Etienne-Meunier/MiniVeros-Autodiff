# Guide to running this project on the server

For an agent that needs to run mini_veros/veros comparisons (`test/*.py`) on
Grid5000 instead of locally. Read this before trying to ssh anywhere.

## What "the server" is here

Grid5000, site **grenoble**, reached through the host alias `agrenoble`
(must already exist in `~/.ssh/config` — this repo doesn't set that up).
The remote copy of this repo lives at `/home/emeunier/code/MiniVeros-Autodiff`.

Sync is handled by [Unison](https://github.com/bcpierce00/unison) via a
generic launcher tool that lives *outside* this repo, at
`~/Desktop/Projets/g5k_launcher` (aliased as the `g5k` shell command). It
reads this repo's `.g5k_config`:

```
model_store_path: MiniVerosAd
project_name: miniverosad
unison_profile: miniverosad_cp_agrenoble
```

## Important: `g5k` is a training-job launcher, not a generic runner

`g5k_launcher` was built for GPU training jobs with wandb sweeps
(`g5k launch <model_cfg> <data_cfg>`, `g5k agent <sweep_id>`). Its resource
string always requests a GPU (`resources_str()` appends `/gpu=N`
unconditionally), and `g5k launch` refuses to run at all without a
`train_script` key in `.g5k_config` — which this repo's config doesn't have.

**Use only these two pieces of `g5k` for this project:**

- `g5k sync code` — one-shot Unison push of the local repo to the remote
  (ignores `.git`, `__pycache__`, `.claude`, `.codegraph`, etc. — see the
  Unison profile at `~/Library/Application Support/Unison/miniverosad_cp_agrenoble.prf`).
- `g5k check --site grenoble` / `g5k log <job_id> --site grenoble` /
  `g5k stop <job_id> --site grenoble` — job status/log/cancel, if you go
  through the batch-submission (`g5k_site.jobs.create`) path instead of a
  plain interactive OAR session. `--site` defaults to `rennes` in the tool —
  **always pass `--site grenoble` explicitly**, or you'll query the wrong site.

**Don't use `g5k launch` / `g5k agent`** — wrong resource model (GPU,
wandb) for this CPU jax comparison matrix.

`model_store_path` in `.g5k_config` is now `MiniVeros-Autodiff`, matching
where every script in `test/` (`generate_matrix_data.py`,
`measure_noise_floor.py`, ...) actually writes (`$STORE/MiniVeros-Autodiff/...`),
so `g5k sync model` should pull results back correctly.

## Running the full matrix as a batch sweep (the usual case)

`test/sweep/run_matrix_oar.sh` is one chunk of the matrix as an OAR batch
job. Submit several, **all with the same run timestamp**:

```
g5k sync code                       # push current local state first
ssh agrenoble
TS=$(date -u +%Y%m%dT%H%M%SZ)
oarsub -l host=1,walltime=2:00:00 -n mvmatrix1 \
  -O ~/oar_logs/matrix_${TS}_c1.out -E ~/oar_logs/matrix_${TS}_c1.err \
  "bash ~/code/MiniVeros-Autodiff/test/sweep/run_matrix_oar.sh $TS acc_basic acc_full ..."
```

The shared timestamp is not cosmetic. `plot_matrix_report.py` resolves
`--timestamp latest` per variant, so chunks stamped at different times leave
the report mixing runs of different lengths — which is how three crashed
variants once came back as the report's only three "passing" rows. Pinning
one timestamp lets `plot_matrix_report.py --strict` verify the whole table
came from a single sweep.

Sizing, measured: an `acc` variant is ~9 min (mini ~110 s + veros ~440 s), a
`global` one ~12 min (~275 s + ~412 s). The full 31-variant matrix is ~5 h of
compute, so ~4 acc or ~3 global variants per chunk fits comfortably inside a
2 h walltime. **Prefer many short chunks**: jobs land in the `p3` queue,
where a 6 h request gets `scheduled_start = no prediction` and can sit
waiting indefinitely while 2 h ones start.

Unison does not carry the executable bit, so invoke the script as
`bash <path>` rather than relying on `./`.

Watch them with `oarstat -u emeunier` and `~/oar_logs/matrix_${TS}_c*.out`;
`g5k sync model` pulls the results back.

## Getting an interactive node instead

For one-off runs and debugging:

```
g5k sync code
ssh agrenoble
oarsub -I -l walltime=2:00:00 --type allow_classic_ssh   # CPU job, no /gpu=
```

Once connected to the allocated node:

```
source ~/.bash_profile
cd /home/emeunier/code/MiniVeros-Autodiff
```

Then activate whatever conda/mamba environment on the remote has
`jax`/`veros`'s dependencies installed — **check this first**
(`mamba env list` or `conda env list`); there's no guarantee it's named the
same as the launcher's own hardcoded training env (`diffusion` — that's for
a different, unrelated project).

## Running things

Same entry points as local, same relative layout (`veros/`, `mini-veros/`,
`test/`):

```
python test/measure_noise_floor.py --variant acc_basic
python test/generate_matrix_data.py --group acc
python test/generate_matrix_data.py                       # full 31-variant matrix
python test/plot_matrix_report.py                          # figures + report.md
pytest test/test_matrix.py -k acc_basic
```

`$STORE` must be defined on the remote (in `~/.bash_profile`, same as
locally) — every data-generating script resolves output paths as
`$STORE/MiniVeros-Autodiff/results/...` and `.../figures/...`. Confirm it's
set (`echo $STORE`) before a long run; if it's unset, scripts fall back to
`~/STORE` instead of erroring, which silently writes somewhere you won't
think to look.

## Getting results back

`g5k sync model` now points at the right path (`model_store_path:
MiniVeros-Autodiff` matches `$STORE/MiniVeros-Autodiff` that the scripts
write to), so it should do a bidirectional sync of `results/`/`figures/`.
If it ever misbehaves, fall back to a plain `scp`/`rsync`/`unison` call
against `$STORE/MiniVeros-Autodiff` directly — these files are small
(`results/*.npz` are hundreds of KB to a few MB).

## Before running anything long

- `pytest test/test_matrix.py` first, on a short horizon
  (`MINIVEROS_TEST_STEPS=5`), to confirm the remote env actually has a
  working `veros` + `jax` before committing walltime to a multi-minute run.
- Long jobs on a shared interactive OAR session die if the ssh session
  drops. Use `tmux`/`screen` on the remote for anything longer than a few
  minutes.
