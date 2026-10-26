# Progress influence under ship motion: illustrative valve simulations

This repository contains the code behind the preliminary analyses in a research proposal by
Soroush Korivand (Mississippi State University) on robots that decide whether their actions still
make progress, why they fail, and when to stop, while working aboard a moving vessel.

**What these results are.** Every number and figure here is produced by numerical simulation of a
deliberately simple, hand-specified toy model. The model parameters were chosen for illustration;
they were not identified from hardware, and no experimental data are used. The simulations test the
logic of the proposed ideas, not their performance on a real robot.

## The toy model

- **Valve:** a single rotating handwheel with friction. It turns only when the applied torque exceeds a
  breakaway level (2.0 in model units). A *seizure* raises that level from 2.0 to 8.0 over 30 s.
- **Robot:** on randomly chosen 0.2-s steps it pushes with torque 4.0; on the other steps it holds
  still (action and no-action trials).
- **Ship motion:** a disturbance torque on the valve, a sinusoid with random phase whose amplitude is
  0.3 in calm sea and 2 to 3.5 in rough sea. The robot measures it with noise.
- **Contact force:** the wrist force depends only on whether the robot pushes, not on whether the
  valve moves, so its "influence" is uninformative about seizure *by construction*.

## Monitors compared (Fig. 2)

| Line | Question it answers |
|---|---|
| Progress monitor | Does the valve turn when the robot pushes? |
| Progress influence, unconditioned | Does the valve move differently when the robot pushes than when it does not? |
| Progress influence, conditioned (proposed) | The same question, after accounting for the measured ship motion |
| Progress predictor (B6 analog) | Does the valve turn less than a model fitted to fault-free episodes predicts? |
| Influence on contact wrench | Does pushing change the force felt at the wrist? |

Alarm thresholds are set on 150 fault-free episodes and evaluated on 150 separate episodes with a seizure.

## Files

| Script | Produces |
|---|---|
| `figs3.py` | Fig. 2 (valve episode; detection delay versus false-alarm rate) and the sequential-test figure and statistics |
| `wrongdir.py` | Wrong-direction case (influence unchanged, advantage changes sign); fraction of pushes that move the valve in rough sea |
| `b6_compare.py` | Progress-predictor baseline (B6 analog) on the same test episodes as Fig. 2b |
| `b6_mismatch.py` | B6 analog and conditioned influence when the test valve is looser or stiffer than the one used for fitting and thresholds |
| `regen_fig3.py` | Attribution example (posterior over five causes after selected probes) |
| `evidence.py` | Decision-latency percentiles for the sequential tests and the power calculations for the planned hardware campaign |
| `figs.py` | Shared plotting style and helper functions |

Outputs are written to `figs/` (figures and JSON statistics) and `logs/` (console output).
The committed outputs are those of a clean run with the random seeds set in the scripts.

## Run

```bash
pip install -r requirements.txt
bash run_all.sh
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, and Matplotlib 3.11.

## Key numbers (5% false-alarm target, 150 test episodes)

| Monitor | Median detection delay after seizure onset |
|---|---|
| Progress influence, conditioned | 11.5 s |
| Progress predictor (B6 analog) | 10.9 s |
| Progress monitor | 22.3 s |
| Progress influence, unconditioned | 41.6 s |

The valve became impossible to turn in any motion phase a median of 21.3 s after onset. On a looser
test valve, the B6 analog slowed to 20.0 s while the conditioned influence stayed at 13.3 s; on a
stiffer test valve, thresholds from the original valve produced false alarms in nearly all fault-free
episodes for both methods.

## License

MIT License (see `LICENSE`).
