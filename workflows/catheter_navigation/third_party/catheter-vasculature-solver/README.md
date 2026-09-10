# catheter-vasculature-solver

Standalone solver package for catheter and guidewire dynamics in vasculature.

## Included solvers

- `XPBDRodSolver`
- `XCathRodSolver`
- `NewtonXPBDRodSolver` (optional runtime dependency)

## Install

```bash
pip install -e .
```

With Newton support:

```bash
pip install -e ".[newton]"
```

## Example import

```python
from catheter_vasculature_solver import RodConfig, XCathRodSolver, XPBDRodSolver
```
