#!/usr/bin/env bash
# Regenerates every figure and number from scratch (about 15 minutes on a laptop).
set -e
mkdir -p figs logs
python3 figs3.py        > logs/log_figs3.txt     # Fig. 2 (first pass), decision-test figure, betting-test statistics
python3 wrongdir.py     > logs/log_wrongdir.txt  # wrong-direction case, rough-sea action effectiveness
python3 b6_compare.py   > logs/log_b6.txt        # progress-predictor baseline (B6 analog) on the Fig. 2b episodes
python3 figs3.py prelim > logs/log_prelim.txt    # redraws Fig. 2 with the B6 analog
python3 b6_mismatch.py  > logs/log_b6_mismatch.txt  # B6 analog and conditioned influence on a different valve
python3 regen_fig3.py   > logs/log_fig3.txt      # attribution example (posterior 0.20 -> 0.60 -> 0.98)
python3 evidence.py     > logs/log_evidence.txt  # decision-latency percentiles and power calculations
echo "Done. Outputs are in figs/ and logs/."
