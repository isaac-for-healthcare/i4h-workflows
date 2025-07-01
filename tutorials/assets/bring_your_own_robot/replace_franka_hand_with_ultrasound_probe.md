# Bring Your Own Robot - Replacing Franka Hand with Ultrasound Probe

This tutorial demonstrates how to replace the Franka Hand with an ultrasound probe to create a robotic ultrasound scanning system in NVIDIA Omniverse.

## Understanding CAD and URDF Formats

### CAD (Computer-Aided Design)
CAD files are 3D models created in specialized software like SolidWorks, Fusion 360, or Blender. They provide detailed geometric information about parts but don't contain information about how parts move relative to each other.

Common CAD formats include:
- `.step` or `.stp` (Standard for Exchange of Product Data)
- `.stl` (Stereolithography)
- `.obj` (Wavefront Object)
- `.iges` or `.igs` (Initial Graphics Exchange Specification)

### URDF (Unified Robot Description Format)
URDF is an XML-based format used in robotics (especially ROS) to describe the physical properties of a robot, including:
- Links (rigid bodies)
- Joints (connections between links with movement constraints)
- Visual appearance
- Collision properties
- Inertial characteristics

A URDF defines not just how a robot looks, but how it moves and interacts with the physical world.

## Converting Files to USD Format

### Converting CAD to USD

NVIDIA Omniverse uses USD (Universal Scene Description) format. To convert your CAD files:

1. **Direct Import**: Isaac Sim supports importing several CAD formats directly:
   - Navigate to `File > Import > Import CAD`
   - Select your CAD file (supported formats include .step, .stp, .iges, .igs)

2. **Using Omniverse Converter**:
   - Install Omniverse Converter
   - Load your CAD file and export to .usd or .usda format

For more details, see the [Omniverse CAD documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_cad-converter/manual.html).

### Converting URDF to USD

To convert URDF to USD for use in Isaac Sim:

1. Use the built-in URDF importer in Isaac Sim:
   - Navigate to `Isaac Utils > URDF Importer`
   - Select your URDF file and configure import settings
   - Click "Import" to generate the USD representation

For detailed steps, refer to the [official documentation on importing URDF](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html).

## Replacing Franka Hand with Ultrasound Probe

### Step 1: Import the Franka Robot
1. In Isaac Sim, load the Franka USD file (either from the built-in assets or by importing a URDF)
2. Examine the robot structure in the Stage panel to identify the hand component

### Step 2: Import the Ultrasound Probe
1. Import your ultrasound probe CAD file and convert it to USD
2. Position it in a convenient location in the scene

   **Example CAD File Provided:**

   An example end-effector CAD file (`HD3C3 Endeffector.step`) is included for you to experiment with. You can find its path like this:

   ```python
   from simulation.utils.assets import robotic_ultrasound_assets as robot_us_assets
   from pathlib import Path
   cad_file_path = Path(robot_us_assets.download_dir) / "Robots" / "Franka" / "End_effector" / "HD3C3 Endeffector.step"
   ```

   **Note:** Please review the NVIDIA Software and Model Evaluation License located in this directory before using the example CAD file.

### Step 3: Remove the Franka Hand
1. In the Stage panel, locate the hand components (usually named something like `/robot/panda_hand`)
2. Right-click and select "Delete" to remove the hand components
3. You may need to remove multiple components including fingers

### Step 4: Position the Ultrasound Probe
1. Move the probe to the location where the hand was removed
2. Use the Transform tools to properly align the probe:
   - Position it at the end of the robot arm
   - Rotate it to the correct orientation
   - Ensure the probe's mounting point aligns with the arm's connection point

### Step 5: Create a Joint to Connect the Probe
1. Select the end effector link of the Franka arm
2. Right-click and select "Create > Physics > Joint"
3. Configure the joint:
   - Set the Parent to the last link of the Franka arm
   - Set the Child to the base of the ultrasound probe
   - Choose the appropriate joint type (usually "Fixed")
   - Set the local poses to align properly

### Step 6: Test the Robot
1. Enter simulation mode to verify the probe moves with the arm:
   - Click the "Play" button in the top toolbar of Isaac Sim
   - Alternatively, press Ctrl+P (or Cmd+P on Mac) to start the simulation
   - The simulation status will change from "Stopped" to "Playing"
2. Test different arm poses to ensure the connection is robust:
   - Use the articulation controller or joint control panel to move the robot
   - Verify that the probe moves naturally with the arm without detaching
   - Check for any unexpected behavior or collisions


## Additional Resources

- [URDF to USD Conversion Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html#tutorial-import-urdf)
- [Omniverse CAD Extension Documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_cad-converter/manual.html)
- [Omniverse Importers and Exporters Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/importers_exporters.html#)
- [Video Tutorial: Training a Robot from Scratch in Simulation, from URDF to OpenUSD](https://www.youtube.com/live/_HMk7I-vSBQ)

## Support

If you encounter any issues during the conversion process, please refer to the tutorial documentation or raise an issue in this repository.
