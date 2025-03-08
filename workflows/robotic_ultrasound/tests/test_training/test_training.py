import os
import shutil
import threading
import time
import unittest

import h5py
import numpy as np
from openpi import train
from policy_runner.config import get_config
from policy_runner.utils import compute_normalization_stats
from training.pi_zero.convert_hdf5_to_lerobot import main as convert_hdf5_to_lerobot


class TestBase(unittest.TestCase):
    """Base class for training tests with common setup and teardown methods."""

    TEST_REPO_ID = "i4h/test_data"

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Determine cache location
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot")
        self.test_data_dir = os.path.join(self.cache_dir, self.TEST_REPO_ID)

        # Setup temporary directories
        self.current_dir = os.path.dirname(__file__)
        self.tmp_assets_dir = os.path.join(self.current_dir, "assets")
        self.tmp_checkpoints_dir = os.path.join(self.current_dir, "checkpoints")
        self.tmp_wandb_dir = os.path.join(self.current_dir, "wandb")
        self.hdf5_data_dir = os.path.join(self.current_dir, "test_data")

        # Create the test_data directory if it doesn't exist
        os.makedirs(self.hdf5_data_dir, exist_ok=True)

        # Create a basic config for testing
        self.test_config = get_config(
            name="robotic_ultrasound_lora", repo_id=self.TEST_REPO_ID, exp_name="test_experiment"
        )
        self.test_prompt = "test_prompt"

        # Flag to allow for cleanup
        self.should_cleanup = False

        # Configure wandb to run in offline mode (no login required)
        os.environ["WANDB_MODE"] = "offline"

    def tearDown(self):
        """Clean up after each test method."""
        if self.should_cleanup:
            # Remove test data directory
            if os.path.exists(self.test_data_dir):
                shutil.rmtree(self.test_data_dir)

            # Remove wandb directory if it exists
            if os.path.exists(self.tmp_wandb_dir):
                shutil.rmtree(self.tmp_wandb_dir)

            # Remove any checkpoints in current directory
            if os.path.exists(self.tmp_checkpoints_dir):
                shutil.rmtree(self.tmp_checkpoints_dir)

            # Remove any assets in current directory
            if os.path.exists(self.tmp_assets_dir):
                shutil.rmtree(self.tmp_assets_dir)

        # Always clean up the test_data directory
        if os.path.exists(self.hdf5_data_dir):
            shutil.rmtree(self.hdf5_data_dir)


class TestConvertHdf5ToLeRobot(TestBase):
    """Test the conversion of HDF5 data to LeRobot format."""

    def setUp(self):
        """Set up test fixtures, including creating a dummy HDF5 file."""
        super().setUp()

        # Create a dummy HDF5 file with the expected structure
        self._create_dummy_hdf5_file()

    def _create_dummy_hdf5_file(self):
        """Create a dummy HDF5 file with 25 steps for testing."""
        num_steps = 50

        # Create the test_data directory if it doesn't exist
        os.makedirs(self.hdf5_data_dir, exist_ok=True)

        # Create a dummy HDF5 file
        hdf5_path = os.path.join(self.hdf5_data_dir, "data_0.hdf5")

        with h5py.File(hdf5_path, "w") as f:
            # Create the root group
            root_name = "data/demo_0"
            root_group = f.create_group(root_name)

            # Create action dataset (num_steps, 6)
            root_group.create_dataset("action", data=np.random.rand(num_steps, 6).astype(np.float32))

            # Create abs_joint_pos dataset (num_steps, 7)
            root_group.create_dataset("abs_joint_pos", data=np.random.rand(num_steps, 7).astype(np.float32))

            # Create observations group
            obs_group = root_group.create_group("observations")

            # Create RGB dataset (num_steps, 2, height, width, 3)
            # Using small 32x32 images to keep the file size small
            rgb_data = np.random.randint(0, 256, size=(num_steps, 2, 32, 32, 3), dtype=np.uint8)
            obs_group.create_dataset("rgb", data=rgb_data)

    def test_convert_hdf5_to_lerobot(self):
        """Test that HDF5 data can be converted to LeRobot format successfully."""
        convert_hdf5_to_lerobot(self.hdf5_data_dir, self.TEST_REPO_ID, self.test_prompt)
        meta_data_dir = os.path.join(self.test_data_dir, "meta")
        data_dir = os.path.join(self.test_data_dir, "data")
        self.assertTrue(os.path.exists(meta_data_dir), f"Meta data directory not created at {meta_data_dir}")
        self.assertTrue(os.path.exists(data_dir), f"Data directory not created at {data_dir}")


class TestNormalizationStats(TestBase):
    """Test the computation of normalization statistics."""

    def test_compute_normalization_stats(self):
        """Test that normalization statistics can be computed successfully."""
        # Compute normalization statistics
        compute_normalization_stats(self.test_config)

        # Check that the stats file was created
        output_path = self.test_config.assets_dirs / self.TEST_REPO_ID
        stats_file = output_path / "norm_stats.json"

        self.assertTrue(os.path.exists(stats_file), f"Stats file not created at {stats_file}")


class TestTraining(TestBase):
    """Test the training process."""

    def test_training_runs_for_one_minute(self):
        """Test that training can run for at least one minute without errors."""
        # Set the flag to True so that the test data is cleaned up after the final test
        self.should_cleanup = True
        # First ensure normalization stats exist
        # Check that the stats file was created
        output_path = self.test_config.assets_dirs / self.TEST_REPO_ID
        stats_file = output_path / "norm_stats.json"

        self.assertTrue(os.path.exists(stats_file), f"Stats file not created at {stats_file}")

        # Start training in a separate thread so we can stop it after a minute
        training_thread = threading.Thread(target=self._run_training)
        training_thread.daemon = True

        # Start timer
        start_time = time.time()
        training_thread.start()

        # Let training run for at least one minute
        time.sleep(60)

        # Check that training ran for at least one minute
        elapsed_time = time.time() - start_time
        self.assertGreaterEqual(elapsed_time, 60, "Training should run for at least one minute")

        # Training should still be running
        self.assertTrue(training_thread.is_alive(), "Training thread should still be running")

    def _run_training(self):
        """Run the training process."""
        try:
            train.main(self.test_config)
        except Exception as e:
            self.fail(f"Training failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
