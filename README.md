# Modular Bose–Clifford Fibre Matchings — Numerical Reproduction

This repository contains a **clean public numerical reproducer** for Tony Newton's manuscript:

**Modular Bose–Clifford Fibre Matchings: Wedge-Local Involutions, Translation-Shell Exchange, and Reflection-Inclusive Finite-Band Bell Optimization**

It contains only equations and finite numerical searches stated in the paper. **It does not contain Enigmai/Newton Solver, private theorem-search code, prompts, attack-generation logic, or other private research machinery.**

## 1. Install Python packages

From a terminal in this folder:

```bash
python -m pip install -r requirements.txt
```

## 2. Reproduce the central values

```bash
python reproduce.py
```

This reproduces the analytic/reference quantities at `r = 0.1`, `omega0 = 0.01`, including approximately:

```text
C_R = 0.96917057543459
B_R = 2.0603750381652
```

## 3. Reproduce the roots

```bash
python reproduce.py --roots
```

This additionally reproduces the shell-transition roots and Bell-bandwidth thresholds.

## 4. Run the exhaustive finite searches

```bash
python reproduce.py --full --json reproduced_results.json
```

The full run includes:

- all `10! = 3,628,800` opposite-parity perfect matchings in the 20-cell translation problem;
- all `140,152` cellwise involutions in the reflection-inclusive 12-cell problem.

The manuscript reference values are approximately:

```text
20-cell translation C_Gamma = 0.9441924536108
12-cell mixed C_Gamma       = 0.96926709992377
12-cell mixed Bell value    = 2.0605115444068
```

The mixed value is the exact optimum **within the stated finite 12-cell grammar**. It is not claimed to be a global optimum over all measurable translation/reflection involutions.

## Numerical method

The program uses deterministic Gaussian quadrature and exhaustive enumeration. No Monte Carlo estimator is used for the headline values.

## Public/private boundary

This repository exists only to make the numerical claims in the manuscript independently reproducible. The private research/discovery system used during development is deliberately excluded.

Copyright © Tony Newton. Public availability is provided for verification and reproducibility of the manuscript results.
