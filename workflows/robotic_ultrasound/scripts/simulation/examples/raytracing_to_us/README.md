### Generate CT samples

#### Download maisi_ct_generative bundle
```bash
python -m monai.bundle download maisi_ct_generative --bundle_dir <path_to_bundle>
```

#### Generate CT samples
```bash
python usd_generation.py --root_dir <path_to_bundle> --ct_samples <num_of_samples> --generate_ct True
```

#### Generate mesh from CT samples
```bash
python usd_generation.py --root_dir <path_to_bundle> --mesh_dir <path_to_save_mesh> --generate_mesh True
```
