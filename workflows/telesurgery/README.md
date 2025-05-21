# Telesurgery

![image](../../docs/source/telesurgery.png)

## System Requirements

- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported.
- NVIDIA Driver Version >= 535
- 50GB of disk space
- Docker & NVIDIA Container Toolkit
- RTI license file
  - Install the license file in `~/rti_license.dat`, or
  - set the `RTI_LICENSE_FILE` environment variable
- VS Code or Cursor AI with devcontainer


### DevContainer

Use `./run vscode` to start a new VS Code or Cursor window for development.

### Running the Application

To run the applications, you may either use Dev Container or use the development container.

Use the following command to enter the development container:

```bash
./run enter
```

### Limitations

*TODO*: The [Dockerfile](./Dockerfile) currently installs a custom build of Holoscan 3.3; update to 3.3 GA when it becomes available.
