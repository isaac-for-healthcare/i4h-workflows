## Holoscan Operators
The Holoscan operators for Isaac-for-Healthcare are located in the `holoscan_i4h/operators/` directory. These operators are modular components designed to be reused across multiple workflows and applications.

### Using the Operators
To import these operators in your code, ensure the root of the i4h-workflows repository is included in your PYTHONPATH. You can do this by running:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/i4h-workflows/
```

You can then import operators as follows:

```python
from holoscan_i4h.operators.realsense.realsense import RealsenseOp
```

### Directory Structure

The initial directory layout under `holoscan_i4h/` is straightforward and is shown below.
The primary directory of interest is the `operators/` folder where the Holoscan operators are located.

```
holoscan_i4h/
├── operators/
│   ├── realsense/
│   ├── clarius_cast/
│   ├── clarius_solum/
│   └── ...
├── CMakeLists.txt
├── README.md
└── __init__.py
```

### Build and Install Folders

Some operators — such as `clarius_solum` — may require a build step. This step is typically triggered automatically as part of the workflow setup process. When this occurs, additional `build/` and `install/` directories will appear:

* `build/`: A temporary directory used by CMake to generate and compile intermediate build artifacts (e.g., object files, build system metadata).
* `install/`: The final output directory for installed components, such as shared libraries (`.so`) and header files (`.h`), ready to be consumed by the operator.

Example layout after build:

```
holoscan_i4h/
├── operators/
├── build/
├── install/
├── CMakeLists.txt
├── README.md
└── __init__.py
```
