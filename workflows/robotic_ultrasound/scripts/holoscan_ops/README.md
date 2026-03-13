# Holoscan Operators

The Holoscan operators for the Robotic Ultrasound workflow are located in this directory: `workflows/robotic_ultrasound/scripts/holoscan_ops/`. These operators are modular components used by the Holoscan apps (Clarius Cast, Clarius Solum, RealSense, etc.).

## Using the operators

Add the **scripts** directory to your `PYTHONPATH` so that the `holoscan_ops` package can be imported (the path must be the directory that contains `holoscan_ops/`):

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/i4h-workflows/workflows/robotic_ultrasound/scripts
```

Then import operators as follows:

```python
from holoscan_ops.operators.realsense.realsense import RealsenseOp
from holoscan_ops.operators.clarius_cast.clarius_cast import ClariusCastOp
from holoscan_ops.operators.clarius_solum.clarius_solum import ClariusSolumOp
from holoscan_ops.operators.no_op.no_op import NoOp
```

### Directory structure

Layout under `holoscan_ops/`:

```text
holoscan_ops/
├── operators/
│   ├── realsense/
│   ├── clarius_cast/
│   ├── clarius_solum/
│   ├── no_op/
│   └── ...
├── CMakeLists.txt
├── README.md
└── __init__.py
```

This directory lives at:

`workflows/robotic_ultrasound/scripts/holoscan_ops/`

### Build and install folders

Some operators (e.g. `clarius_solum`, `clarius_cast`) require a build step, usually run via the workflow or `./i4h`. After building, `build/` and `install/` appear under this directory:

- **`build/`** – CMake build artifacts (object files, build metadata).
- **`install/`** – Installed components (e.g. shared libraries under `install/lib/clarius_cast/`, `install/lib/clarius_solum/`) used by the Python operators.

Example layout after build:

```text
holoscan_ops/
├── operators/
├── build/
├── install/
│   └── lib/
│       ├── clarius_cast/
│       └── clarius_solum/
├── CMakeLists.txt
├── README.md
└── __init__.py
```

For running the Holoscan apps, see [Holoscan Apps README](../holoscan_apps/README.md) and the main [Robotic Ultrasound README](../../README.md).
