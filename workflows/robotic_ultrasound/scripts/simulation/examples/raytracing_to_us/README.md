### Generate CT samples

#### Download Sample Dataset

```bash
wget -P <path_to_save_dataset> https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
tar -xvf <path_to_save_dataset>/Task09_Spleen.tar
```

#### Download vista3d bundle and run inference

We use the vista3d bundle to generate segmentation labels for the sample dataset.

```bash
python -m monai.bundle download vista3d --bundle_dir <path_to_bundle>
```

Then, in the bundle directory, run the following command to generate segmentation labels.

```bash
python -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json']" --input_dir <path_to_save_dataset>/Task09_Spleen/imagesTr --output_dir <path_to_save_dataset>/Task09_Spleen/predsTr --separate_folder False
```


#### Generate USD from segmentation labels
```bash
python usd_generation.py --seg_dir <path_to_save_dataset>/Task09_Spleen/predsTr
```
