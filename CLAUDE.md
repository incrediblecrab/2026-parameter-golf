# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and GitHub Copilot CLI when working in this repository.

This is a public fork of `openai/parameter-golf` kept as a study collection for the Parameter Golf challenge (train the best language model under 16,000,000 bytes and 10 minutes): the upstream checkout at commit `f5c0793`, four extra PR submission directories copied at pinned revisions, provenance records and a curated reading guide in `README.md`. It is a source study, not a reproduction: no GPU jobs have been run and no training data or checkpoints are stored. The upstream README is preserved as `UPSTREAM_README.md`.

## Commands

```bash
python3 scripts/check_readme_claims.py   # recomputes every score, byte count and time quoted in README.md from records/; exits 1 on a mismatch
```

Run it after any change to `README.md` or `records/`; it checks that the quoted numbers match the authors' logs, not that the code reproduces them. Everything else (each entry's `train_gpt.py`, `train_gpt.py` and `train_gpt_mlx.py` at the root, `scripts/run_1868_rerun.py` for 8×H100, `measure_variance_ratio.py` for trained checkpoints) needs the upstream environment from `requirements.txt` plus CUDA and the per-entry dependencies, and often a GPU cluster; do not launch training or download datasets unless the owner asks. `train_gpt_mlx.py` is the only Apple Silicon entry point and is a starter, not a port of the leading models.

## Structure

`records/track_10min_16mb/` holds accepted leaderboard entries and `records/track_non_record_16mb/` non-record studies, one dated folder per submission with its own README, training script, logs and `submission.json`. These are the authors' files: leave them byte-for-byte as submitted. `model_sources.json` maps each featured model to its directory, source revision and entry point, and the README's provenance table lists the PRs copied in (#2135, #2014, #1953, #1110) with their pinned revisions; keep the two in step when adding one. `data/` holds the upstream dataset and tokenizer helpers, `paper/` the DG Attention paper, and `results/pr1868_rerun_compliance/` a patched script for a compliance re-run. `LICENSE` and `THIRD_PARTY_NOTICES.md` carry the upstream licensing.

Claims about our own submission stay as the README states them: it has not been identified in this checkout, so nothing here diagnoses why it underperformed.
