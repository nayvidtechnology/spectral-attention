# Experiments

- **runs/**: raw outputs per run (never edited).
- **configs/**: YAML configs for reproducible runs.
- **promote/**: candidates for docs; move only validated results.

### Run Naming
`YYYY-MM-DDThhmmZ_<shorttag>` e.g., `2025-09-07T1730Z_seq4k_fft`.

### Metrics schema
We log newline-delimited JSON records. Common events:

- Throughput/memory:
	`{"event":"throughput","kind":"spectral|vanilla","seq":int,"tokens_per_s":float,"ms_per_it":float,"peakMB":float}`

- Training (task-agnostic):
	`{"event":"train","step":int,"loss":float,"ppl":float?,"acc":float?,"kind":"spectral|vanilla","seq":int}`

- Validation (task-agnostic):
	`{"event":"val","val_loss":float,"val_ppl":float?,"val_acc":float?,"kind":"spectral|vanilla","seq":int}`

Where `ppl` fields appear for LM tasks, and `acc` fields appear for classification tasks.
