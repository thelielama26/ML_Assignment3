# ML Assignment 3 - CIFAR10 (MLP, CNN, Autoencoders, SVM, Clustering)

Name:
Roll number:
Repository link:
Final commit hash:

## Layout
- data/split/   saved stratified 45k/5k split (indices + seed record)
- configs/      one config file per experiment
- src/q1..q4/   code for each question
- scripts/figures/  scripts that regenerate every figure/table
- logs/         final training/eval logs per question
- results/      tables and figures used in the report
- checkpoints/  saved models (not committed if large)
- report/       CVPR LaTeX source and PDF

## Environment
(environment file: add here)

## Reproduction commands
(exact commands, in order, to reproduce every reported number)

## Reuse log
| Model / code / result | First introduced in | Reused in |
|---|---|---|

## Pre-registered decisions (write BEFORE final runs)
- Q1a fairness criterion:
- Q2a fairness criterion:
- Q2b expected result:
- Q4 subset (if used):

## Q2 pre-registration (written before running Q2)
Reused from Q1a: models `cnn` and `mlp_matched` (definitions in src/q1/models.py), Q1a hyper-parameters
(results/tables/q1a_hparams.json) and Q1a checkpoints (used as the CLEAN condition).
- Q2a fairness criterion: parameters matched within ~1%, same optimiser / lr / epochs / batch size, no augmentation,
  selection by best validation loss, 3 seeds. Proposed reason for the difference: spatial inductive bias
  (local connectivity + weight sharing), not capacity.
- Q2b transform: ONE fixed random permutation of the 1024 pixel positions (same for all channels, images, splits).
  Expected: MLP accuracy ~unchanged (within ~1-2 pts); CNN drops substantially, to roughly the MLP's level or below.
- Q2c explanations: (E1) CNN relies on colour / low-order local statistics; (E2) CNN relies on the global arrangement
  of local features. Arms: `gray` (colour removed) and `patch8` (16 patches of 8x8 shuffled, fixed permutation).
  Expected: if E2, patch8 hurts the CNN much more than gray; if E1, gray hurts more than patch8. My expectation: E2.

## Q3 pre-registration (written before running Q3)
Reused from Q1a: fixed classifier = `cnn` seed 0 (checkpoints/q1a/cnn_s0.pt), never fine-tuned.
- Q3a: conv AE, bottleneck d in {16, 64, 256} (only the two bottleneck linear layers differ); MSE in [0,1] pixel space;
  Adam lr 1e-3, 40 epochs, batch 128, same init seed. Selection if reconstruction is the only criterion:
  lowest VALIDATION MSE. Caveat: that criterion trivially favours the largest bottleneck.
- Q3b: corruption = additive Gaussian noise sigma=0.2 (clipped to [0,1]); DAE has the selected clean AE's architecture,
  trained on freshly corrupted train images -> clean targets, no labels. Val/test noise is fixed (seeded).
  Expectation: corrupted images hurt the CNN a lot; DAE recovers most (not all) of the lost accuracy; the clean AE
  (never trained on noise) recovers less.
- Q3c: frozen encoders, linear + kNN probes (hyper-parameters chosen on validation). Expectation: reconstruction error
  improves monotonically with d, but class information saturates earlier, so the two quantities do NOT fully agree.
