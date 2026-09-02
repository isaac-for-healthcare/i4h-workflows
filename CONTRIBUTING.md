# Contributing to NVIDIA Isaac for Healthcare

## License Guidelines

- Make sure that you can contribute your work to open source. Verify that no license and/or patent conflict is introduced by your code. NVIDIA is not responsible for conflicts resulting from community contributions.

- We require community submissions under the Apache 2.0 permissive open source license, which is the [default for Isaac for Healthcare](./LICENSE).

- We require that members [sign](#signing-your-contribution) their contributions to certify their work.

### Coding Guidelines

- All source code contributions must strictly adhere to the Isaac for Healthcare coding style.

### Signing Your Contribution

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

- Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```text
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
   pre-commit run --all-files
   ```

## Running Tests

Set up the independently locked component environments, then run the repository test dispatcher:

```bash
./setup.sh
python scripts/run_tests.py
```

For only the fastest workflow-contract subset:

```bash
python scripts/run_tests.py --suite light
```

To run the same CPU suites and coverage report used by GitHub Actions:

```bash
python scripts/run_tests.py --suite ci --coverage
```

The generated `coverage.xml` reports coverage for CPU-testable repository code. Arena and simulator integration are validated separately by the GPU smoke suite.

Suites whose project depends on an internal component checkout are skipped, and named at the end of the run, when `./setup.sh` has not cloned it. GitHub Actions clones the patient digital twin from `main` when the `I4H_INTERNAL_READ_TOKEN` secret is present, so the patient-twin suite runs there on internal pull requests. Forks receive no secrets, so it is skipped on those and has to be run locally against a checkout.

On a configured GPU host, run the recorded headless workflow smokes used by Blossom:

```bash
python scripts/run_tests.py --suite gpu
```

## Reporting issues

Please open a [Issue Request](https://github.com/isaac-for-healthcare/i4h-workflows/issues) to request an enhancement, bug fix, or other change in Isaac for Healthcare.
