## Pre-commit

To install `pre-commit`, run the following command:
```bash
pip install pre-commit
```

To check your code before committing, run the following command:

```bash
pre-commit run --all-files
```
This will only check the changes and will not modify the code.

To run autofix, run the following command:

```bash
pre-commit run -c tools/premerge-autofix.yaml --all-files
```

This will fix the linting errors and formatting errors.


## Test

```bash
# Optional: Install dependencies for CI pipelines
python tools/install_deps.py

# Run all tests
python tools/run_all_tests.py
```
