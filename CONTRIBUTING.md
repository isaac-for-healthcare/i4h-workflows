# Contributing to NVIDIA Isaac for Healthcare

## License Guidelines

- Make sure that you can contribute your work to open source. Verify that no license and/or patent conflict is introduced by your code. NVIDIA is not responsible for conflicts resulting from community contributions.

- We require community submissions under the Apache 2.0 permissive open source license, which is the [default for Isaac for Healthcare](./LICENSE).

- We require that members [sign](#signing-your-contribution) their contributions to certify their work.

### Coding Guidelines

- All source code contributions must strictly adhere to the Isaac for Healthcare coding style.

### Signing Your Contribution

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

* Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality. To set up pre-commit:

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. To check your code before committing:
   ```bash
   pre-commit run --all-files
   ```

3. To automatically fix linting and formatting errors:
   ```bash
   pre-commit run -c tools/premerge-autofix.yaml --all-files
   ```

## Running Tests

Before submitting your contribution, ensure all tests pass in the workflow(s) you contributed:

```bash
# Install dependencies
python tools/install_deps.py --workflow <workflow_name>

# Download required assets
i4h-asset-retrieve

# Set up your RTI license, skip if you are only running tests for robotic_surgery
export RTI_LICENSE_FILE=<path to your RTI license file>

# Run all unit tests
python tools/run_all_tests.py --workflow <workflow_name>

# Run all integration tests
python tools/run_all_tests.py --integration
```

## Reporting issues

Please open a [Issue Request](https://github.com/isaac-for-healthcare/i4h-workflows/issues) to request an enhancement, bug fix, or other change in Isaac for Healthcare.
