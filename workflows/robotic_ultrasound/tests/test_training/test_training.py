import os
import shutil
import threading
import time
import unittest

from policy_runner.config import get_config
from policy_runner.utils import compute_normalization_stats
from training.pi_zero.convert_hdf5_to_lerobot import main as convert_hdf5_to_lerobot

from openpi import train


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

        # Create a basic config for testing
        self.test_config = get_config(
            name="robotic_ultrasound_lora", repo_id=self.TEST_REPO_ID, exp_name="test_experiment"
        )
        self.test_prompt = "test_prompt"

        # Flag to allow for cleanup
        self.should_cleanup = False

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


class TestConvertHdf5ToLeRobot(TestBase):
    """Test the conversion of HDF5 data to LeRobot format."""

    def test_convert_hdf5_to_lerobot(self):
        """Test that HDF5 data can be converted to LeRobot format successfully."""
        convert_hdf5_to_lerobot(self.hdf5_data_dir, self.TEST_REPO_ID, self.test_prompt)


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
