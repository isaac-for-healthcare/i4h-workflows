# Bring Your Own Patient

This guide helps you convert your own CT or MRI scans into USD (Universal Scene Description) files for 3D visualization.

## Overview
You can use your own medical imaging data (CT or MRI scans) and corresponding segmentation masks to create 3D models of organs and anatomical structures in USD format. This format is widely used in 3D visualization and can be loaded into various applications supporting USD.

## Preparing Model Objects and USD

### Requirements
Before starting, ensure you have:
- Your medical imaging data and corresponding segmentation masks in supported formats:
  - NIFTI (.nii, .nii.gz), NRRD (.nrrd), Single-series DICOM images (.dcm)
  - Or use MAISI to generate synthetic data
- MONAI framework installed
- Required dependencies as specified in the tutorial

### Steps
For a detailed walkthrough of the conversion process, please refer to the [MONAI Omniverse Integration Tutorial](https://github.com/Project-MONAI/tutorials/blob/main/modules/omniverse/omniverse_integration.ipynb). This comprehensive tutorial demonstrates:

- How to load and preprocess medical imaging data
- Converting segmentation masks to 3D meshes
- Exporting the results to USD format
- Visualizing the generated 3D models

### Updating the Assets in the Simulation

Currently, the robotic ultrasound workflow uses the assets in [I4H Asset Catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog). To replace the assets with your own, you need to modify the [environment configuration file](../../../workflows/robotic_ultrasound/scripts/simulation/exts/robotic_us_ext/robotic_us_ext/tasks/ultrasound/approach/config/franka/franka_manager_rl_env_cfg.py) to update the `usd_path` for `organs` in `RoboticSoftCfg`, as well as the `mesh_dir` in [ultrasound_raytracing.py](../../../workflows/robotic_ultrasound/scripts/simulation/examples/ultrasound_raytracing.py).

### Assets from different sources

If you generate the mesh and USD files from the different source (e.g. mesh from CT scan but USD from public human body 3D model), you need to align the meshes to achieve realistic simulation results.

- USD model follows the [USD convention](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/reference_material/reference_conventions.html#usd-axes).
  - It is the `organ` frame in the [environment configuration file](../../../workflows/robotic_ultrasound/scripts/simulation/exts/robotic_us_ext/robotic_us_ext/tasks/ultrasound/approach/config/franka/franka_manager_rl_env_cfg.py)

- Mesh object for the organs
  - It could be derived from CT/MRI scans, MAISI or 3D models from other sources.
  - We assume the organ mesh files share the same coordinate system. If not, you need to use 3D visualization software to align them.

- Approximate align meshes and model by offsetting and rotating the axes:

#### Step 1: Compute the offset needed to center the organ meshes

Treat the organ meshes as a whole, and find the offset to center the organ meshes. So that the origin of the organ meshes is the center of the mass of the organ meshes.

```math
v_{offset} = \{x_{center},\ y_{center},\ z_{center}\}
```

#### Step 2: Find the rotation matrix to align the USD axes with the organ axes
Here we assume the organ meshes has Superior-Inferior-Left-Right-Anterior-Posterior (SI-LR-AP) axes. We used those axis to align the organ meshes with the exterior model.

We place the body model in both the mesh and USD coordinate systems. Ribs and spines are rendered to show the orientation.
![image](../../../docs/source/transformation.png)

It need to be noted that this placement of the body model is based on how the USD model is created. You will need to make sure SI-LR-AP axes are generally aligned.


Find the basis vector in the coordinate system representing the internal organ meshes
```math
\vec{v}_{mesh_{lr}} = (-1,\ 0,\ 0)
```
```math
\vec{v}_{mesh_{ap}} = (0,\ 1,\ 0)
```
```math
\vec{v}_{mesh_{si}} = (0,\ 0,\ -1)
```

Find the basis vector in the USD coordinate system representing placement of the exterior model
```math
\vec{v}_{usd_{lr}} = (-1,\ 0,\ 0)
```
```math
\vec{v}_{usd_{ap}} = (0,\ 0,\ -1)
```
```math
\vec{v}_{usd_{si}} = (0,\ -1,\ 0)
```

Find the rotation matrix to map USD world coordinate system to the organ mesh coordinate system
```math
R_{mesh \rightarrow usd} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & -1 \\
0 & 1 & 0
\end{bmatrix}
```

#### Step 3: Convert the rotation matrix to quaternion


```math
q_{mesh \rightarrow usd} = \text{quat\_from\_matrix}(R_{mesh \rightarrow usd})
```
In the default setup above, the quaternion is `(0.7071, 0.7071, 0, 0)` and offset is `(0, 0, 0)` in `mesh_to_organ_transform`.


### Alignment of Assets from one source

Coming soon.

## Support
If you encounter any issues during the conversion process, please refer to the tutorial documentation or raise an issue in this repository.
