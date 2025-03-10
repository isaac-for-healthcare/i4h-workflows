# i4h-workflows

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

export RTI_LICENSE_FILE=<path to your RTI license file>
ls $RTI_LICENSE_FILE

# Optional: Download the assets
git clone git@github.com:isaac-for-healthcare/i4h-asset-catalog.git
cd i4h-asset-catalog
git checkout mz/asset_mgmt  # FIXME: remove this after the asset catalog is merged
pip install -e .
# NOTE: for the first time, make sure to run this command with a display. # Otherwise, you will not be able to authenticate in a web browser.
i4h-asset-retrieve

# Run all tests
python tools/run_all_tests.py
```
