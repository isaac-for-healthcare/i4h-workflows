# Bring Your Own Medical Images

This guide helps you convert your own CT or MRI scans into USD (Universal Scene Description) files for 3D visualization.

## Overview
You can use your own medical imaging data (CT or MRI scans) and corresponding segmentation masks to create 3D models of organs and anatomical structures in USD format. This format is widely used in 3D visualization and can be loaded into various applications supporting USD.

## Tutorial Reference
For a detailed walkthrough of the conversion process, please refer to the [MONAI Omniverse Integration Tutorial](https://github.com/Project-MONAI/tutorials/blob/main/modules/omniverse/omniverse_integration.ipynb). This comprehensive tutorial demonstrates:

- How to load and preprocess medical imaging data
- Converting segmentation masks to 3D meshes
- Exporting the results to USD format
- Visualizing the generated 3D models

## Requirements
Before starting, ensure you have:
- Your medical imaging data and corresponding segmentation masks in supported formats:
  - NIFTI (.nii, .nii.gz), NRRD (.nrrd), Single-series DICOM images (.dcm)
  - Or use MAISI to generate synthetic data
- MONAI framework installed
- Required dependencies as specified in the tutorial

## Aligning the internal organ meshes with the exterior model from different assets

If you have different assets for the patients, e.g. model for the exterior of the body and mesh models for the internal organs, you can use the following transformations to combine them.

### Model for the outside of the body
Exterior model asset follows the [USD convention](https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html#usd-axes).

### Models for the inside organs

The models can be meshe files of different organs organized with relative positions. It could be derived from CT/MRI scans, MAISI or 3D models from other sources.

### Steps to approximate align meshes and model

#### Compute the offset needed to center the organ meshes

Treat the organ meshes as a whole, and find the offset to center the organ meshes. So that the origin of the organ meshes is the center of the mass of the organ meshes.

```math
v_{offset} = \{x_{center}, y_{center}, z_{center}\}
```

#### Find the rotation matrix to align the USD axes with the organ axes
Here we assume the organ meshes has Superior-Inferior-Left-Right-Anterior-Posterior (SI-LR-AP) axes. We used those axis to align the organ meshes with the exterior model.

1. Find the basis vector in the coordinate system representing the internal organ meshes
```math
\vec{v}_{mesh_lr} = (1, 0, 0)
\vec{v}_{mesh_ap} = (0, 1, 0)
\vec{v}_{mesh_si} = (0, 0, 1)
```

2. Find the basis vector in the USD coordinate system representing placement of the exterior model
```math
\vec{v}_{usd_lr} = (-1, 0, 0)
\vec{v}_{usd_ap} = (0, 0, -1)
\vec{v}_{usd_si} = (0, -1, 0)
```

3. Find the rotation matrix to map USD world coordinate system to the organ mesh coordinate system
```math
R_{mesh \rightarrow usd} = \begin{bmatrix}
-1 & 0 & 0 \\
0 & 0 & -1 \\
0 & -1 & 0
\end{bmatrix}
```

4. To finish the transformation, we have
```math
T_{mesh \rightarrow usd} = \begin{bmatrix}
R_{mesh \rightarrow usd} & v_{offset} \\
0 & 1
\end{bmatrix}
```

We offer a helper function to load the transformation (FIXME: add link). To bring your sets of exterior model and meshes, you will need to convert the rotation matrix to quaternion and pass it with the offsets:
```math
q_{mesh \rightarrow usd} = \text{quat\_from\_matrix}(R_{mesh \rightarrow usd})
```
In the setup above, the quaternion is `[0.0000,  0.0000,  0.7071, -0.7071]`.


### Alignment with organ meshes that abide by NIFTI coordinate system

TBD

## Support
If you encounter any issues during the conversion process, please refer to the tutorial documentation or raise an issue in this repository.
