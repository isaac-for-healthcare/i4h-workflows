# 🏥 Medical Data Conversion

## 🎯 Learning Objectives

By the end of this chapter, you will be able to:
- [ ] Use MAISI CT to generate synthetic CT data
- [ ] Convert CT data to USD format
- [ ] Understand the value of synthetic data for building autonomous healthcare robots

---

## 🚀 Benefits of Synthetic Data Generation (SDG)

| Benefit | Medical Imaging | Robotics Technologies |
|---------|----------------|---------------------|
| **🔍 Addressing Data Scarcity** | Medical imaging data, especially for rare diseases or edge cases, is often scarce or expensive to collect. Synthetic data fills these gaps by generating realistic images, improving data diversity and volume for training AI models. | Robotics applications often require vast, diverse datasets for training machine learning models in tasks like object recognition or motion planning. Synthetic data provides scalable, customizable data for these needs. |
| **🔒 Enhancing Privacy** | Synthetic data is not linked to real individuals and thus avoids patient privacy concerns and regulatory hurdles. This enables wider data sharing and collaboration without violating laws like HIPAA or GDPR. | In scenarios where sensor or camera data is sensitive, synthetic data can be shared more freely for cross-team or cross-institution advancement. |
| **💰 Cost and Efficiency** | Creating and annotating real medical images is time-consuming and costly. Synthetic data can be rapidly generated at lower cost, expediting the development and validation of AI tools. | Generating and annotating real-world robotics data can be prohibitively expensive. Synthetic data circumvents this by allowing efficient creation and labeling of diverse datasets. |
| **⚖️ Reducing Bias** | By generating data for underrepresented populations or rare conditions, synthetic datasets can help reduce bias in AI models, leading to fairer, more generalizable healthcare solutions. | Exposure to a broad spectrum of synthetic scenarios enhances robots' adaptability to new or unseen real-world conditions. This is crucial for applications like assistive robotics or autonomous navigation. |
| **⚡ Accelerating Innovation** | Synthetic data is used to train, validate, and benchmark AI models, speeds up clinical trials simulation, and supports medical education by providing diverse case material. | Robotic systems can be tested and trained in photorealistic or highly variable virtual environments, including edge cases that are rare or hazardous in the physical world, increasing safety and robustness. |

### 🔧 Simulation Benefits

Simulation environments reduce development time and cost by enabling rapid prototyping and testing of algorithms and designs entirely in a virtual setting—eliminating the need to build and modify early physical prototypes. This approach allows software to be developed and iterated quickly, accelerates the engineering timeline, and lowers expenses related to hardware and materials.

**Software-in-the-loop (SIL)** testing lets developers validate control algorithms in a fully simulated environment, allowing fast, low-risk iterations. **Hardware-in-the-loop (HIL)** testing connects real hardware to simulated scenarios, detecting hardware-specific issues and increasing system reliability before full deployment—all while reducing the need for costly prototype builds.

---

## 🧠 MAISI CT: Foundational CT Volume Generation Model

Patient anatomy examples were generated using the **MAISI foundational CT volume generation model**, which leverages generative AI to create high-quality, diverse synthetic CT data for medical imaging research and development. MAISI CT helps address data scarcity and privacy challenges in healthcare AI by providing realistic, customizable anatomical datasets.

### 📚 Resources
- [📖 Overview Blog: Addressing Medical Imaging Limitations with Synthetic Data Generation](https://developer.nvidia.com/blog/addressing-medical-imaging-limitations-with-synthetic-data-generation/)
- [📄 MAISI CT Paper (arXiv)](https://arxiv.org/html/2409.11169v1)
- [🔧 MAISI Nvidia Inference Microservice (NIM)](https://build.nvidia.com/nvidia/maisi)
- [📦 Project-MONAI/models/maisi_ct_generative (Model Zoo)](https://github.com/Project-MONAI/model-zoo/tree/dev/models/maisi_ct_generative)

---

## 🛠️ Run MAISI CT Pipeline Locally with MONAI Model Zoo

### 1️⃣ **Clone the repo and install maisi_ct_generative**

Follow the steps in the [official repository](https://github.com/Project-MONAI/model-zoo/tree/dev/models/maisi_ct_generative) to clone and install the model. The following modifications were tested on git hash: `05067dce4db8fcb87dc31e7fa510c494959230ea`

```bash
pip install "monai[fire]"
python -m monai.bundle download "maisi_ct_generative" --bundle_dir "bundles/"
```

> **💡 Tip:** The standard model requires a selection of anatomical features, though skin is not one of them. For our purposes, we can simply uncomment the filter function in `bundles/maisi_ct_generative/scripts/sample.py`. This will save all labels used during the data generation.

```bash
# synthetic_labels = filter_mask_with_organs(synthetic_labels, self.anatomy_list)
```

### 2️⃣ **Adjust the config to have an empty anatomy_list**

- Copy the inference script and modify it with the below instructions
- Edit the configuration file (e.g., `configs/inference_all.json`) and set `anatomy_list` to an empty list (`[]`)
- You may need to adjust additional parameters in the config to fit the model on your GPU. The file in `utils/config/inference_all.json` was used to generate the sample CTs for this course
- This ensures that all labels will be returned in the output

### 3️⃣ **Run MAISI from the MONAI Model Zoo**

```bash
python -m monai.bundle run --config_file configs/inference_all.json
```

### 4️⃣ **Visualize generated CT Data**

Install Slicer SDK or another application to view the CT data and labelmap.

![MAISI_CT_data](../../../docs/source/Slicer_view_SDG.png)

---

## 🤔 Why Convert a CT Dataset from NIfTI or DICOM to USD?

Medical imaging data, such as CT scans, are typically stored in formats like **NIfTI (.nii, .nii.gz)** or **DICOM (.dcm)**. These formats are well-suited for clinical and research workflows, as they efficiently store volumetric (3D) data and associated metadata. However, they are not directly compatible with 3D simulation, visualization, or robotics platforms like NVIDIA IsaacSim.

**Universal Scene Description (USD)** is a powerful, extensible 3D file format developed by Pixar and widely adopted in the visual effects, animation, and simulation industries. USD is designed for efficient scene representation, asset interchange, and real-time rendering.

### 🎯 Key Reasons for Conversion:

- **🔄 Interoperability with Simulation Platforms:**
  IsaacSim and other robotics/graphics tools natively support USD for importing, manipulating, and rendering 3D assets. Converting medical data to USD enables seamless integration into these environments.

- **🔲 Mesh Representation:**
  NIfTI and DICOM store volumetric data (voxels), but simulation and visualization platforms require surface meshes (e.g., OBJ, STL, or USD) to represent anatomical structures. The conversion process extracts and processes these meshes from the volumetric data.

- **⚡ Efficient Rendering and Manipulation:**
  USD supports hierarchical scene graphs, material definitions, and efficient rendering pipelines, making it ideal for interactive applications, simulation, and digital twin workflows.

- **📊 Rich Metadata and Structure:**
  USD allows for the inclusion of semantic labels, hierarchical organization, and physical properties, which are essential for robotics, AI training, and advanced visualization.
  > 📚 **Learn more about USD:** [NVIDIA OpenUSD Learning Path](https://www.nvidia.com/en-us/learn/learning-path/openusd/)

### 📋 Summary

Converting CT datasets from NIfTI or DICOM to USD is necessary to:
- ✅ Enable use in simulation and robotics platforms like IsaacSim
- ✅ Transform volumetric medical data into usable 3D surface meshes
- ✅ Leverage the advanced features and performance of the USD ecosystem

---

## 🔄 Conversion to USD Format
The tool implements the complete conversion workflow:
1. **NRRD → NIfTI:** Format conversion for medical data
2. **NIfTI → Mesh:** Surface extraction with smoothing and reduction
3. **Mesh → USD:** Final format suitable for IsaacSim and 3D visualization

## 🛠️ Setup NRRD to USD Converter Tool

This project includes an `environment.yml` file for easy conda environment setup. To set up the environment and install all required dependencies for the CT-to-USD conversion workflow, you have two options:

### Option 1: Use Existing i4h-workflows Conda Environment

If you already have the i4h-workflows conda environment set up, you can install the dependencies directly:

```bash
# Activate the existing i4h environment
conda activate i4h

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### Option 2: Create a New Dedicated Conda Environment

Create a dedicated environment for the CT-to-USD converter:

```bash
# Create and activate new environment
conda env create -f environment.yml
conda activate ct-to-usd-converter
```



### 1️⃣ **NRRD to NIfTI Conversion**
- Convert segmentation files from NRRD to NIfTI format

### 2️⃣ **NIfTI to Mesh Processing**
- Groups 140 labels into 17 anatomical categories
- Creates separate OBJ files for each organ
- Applies smoothing and mesh reduction
- Supports detailed anatomical structures including:
  - 🫀 **Organs:** Liver, Spleen, Pancreas, etc.
  - 🦴 **Skeletal system:** Spine, Ribs, Shoulders, Hips
  - 🩸 **Vascular system:** Veins and Arteries
  - 💪 **Muscular system:** Back muscles

### 3️⃣ **Mesh to USD Conversion**
- Converts the processed meshes to USD format
- Final USD files can be imported into IsaacSim

---


### 🚀 Usage

**Basic Command:**
```bash
# Make sure your conda environment is activated first
conda activate ct-to-usd-converter  # or conda activate i4h

# Then run the converter
python utils/converter.py /path/to/your/ct_folder
```


### 📁 Output Structure

The converter generates:
```
output/
├── nii/           # Intermediate NIfTI files
├── obj/           # Intermediate mesh files
└── all_organs.usd # Final USD file
```

### 🏥 Supported Anatomical Structures

The tool processes 140 labels grouped into 17 categories including:

- **🫀 Organs:** Liver, Spleen, Pancreas, Heart, Gallbladder, Stomach, Kidneys
- **🍽️ Digestive:** Small bowel, Colon
- **🦴 Skeletal:** Spine, Ribs, Shoulders, Hips
- **🩸 Vascular:** Veins and Arteries
- **🫁 Respiratory:** Lungs
- **💪 Muscular:** Back muscles

---

## 🎮 Load CT-derived data into IsaacSim

### 📋 Steps:

1. **🚀 Launch IsaacSim again**
2. **📂 Find the path of your I4H-assets**
3. **📁 Open the CT-derived USD file**


---


## 🎉 Summary

This chapter has covered the essential aspects of synthetic medical data generation and conversion, providing you with the tools and knowledge to work with MAISI CT and convert medical imaging data to USD format for use in IsaacSim and other robotics platforms.
