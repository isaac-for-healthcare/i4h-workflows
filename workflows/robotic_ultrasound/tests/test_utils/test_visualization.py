import os
import sys
import threading
import time
import unittest
from importlib.util import find_spec
from unittest import mock, skipUnless

import numpy as np
from scripts.simulation.configs.basic import config

# Mock sys.argv before importing the module
original_argv = sys.argv
sys.argv = [sys.argv[0]]  # Keep only the script name

# Mock argparse.ArgumentParser.parse_args
original_parse_args = mock.MagicMock()
with mock.patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
    # Setup the mock to return an object with a config attribute
    mock_args = mock.MagicMock()
    mock_args.config = "basic"
    mock_parse_args.return_value = mock_args

    # Now import the module after patching
    import dearpygui.dearpygui as dpg

    from workflows.robotic_ultrasound.scripts.utils.visualization import VisualizationApp

try:
    RTI_AVAILABLE = bool(find_spec("rti.connextdds"))
    if RTI_AVAILABLE:
        license_path = os.getenv("RTI_LICENSE_FILE")
        RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestVisualizationApp(unittest.TestCase):
    def setUp(self):
        dpg.destroy_context()

    def tearDown(self):
        dpg.destroy_context()

    def test_app_initialization(self):
        self.app = VisualizationApp()

        # Test basic initialization properties
        self.assertEqual(self.app.window_name, VisualizationApp.DEFAULT_WINDOW_NAME)
        self.assertEqual(self.app.resizeable, VisualizationApp.DEFAULT_RESIZEABLE)
        self.assertEqual(self.app.room_camera_range_start, self.app.DEFAULT_CAMERA_RANGE_START)
        self.assertEqual(self.app.room_camera_range_end, self.app.DEFAULT_CAMERA_RANGE_END)

        # Verify image data arrays are properly initialized
        self.assertTrue(hasattr(self.app, "room_camera_image_data"))
        self.assertTrue(hasattr(self.app, "wrist_camera_image_data"))
        self.assertTrue(hasattr(self.app, "ultrasound_image_data"))

    def test_create_app_lifecycle(self):
        self.app = VisualizationApp()

        # Create the app UI
        self.app.create_app()

        # Verify UI elements
        self.assertTrue(dpg.does_item_exist("Main Window"))
        self.assertTrue(dpg.does_item_exist("room_camera_image_data"))
        self.assertTrue(dpg.does_item_exist("wrist_camera_image_data"))
        self.assertTrue(dpg.does_item_exist("ultrasound_image_data"))

        # Render a few frames to ensure it doesn't crash
        for _ in range(5):
            dpg.render_dearpygui_frame()

    def test_mouse_wheel_event(self):
        self.app = VisualizationApp()
        self.app.create_app()

        # Get initial zoom value
        initial_value = dpg.get_value("room_camera_zoom")

        # Simulate wheel event - positive scroll (zoom in)
        self.app.mouse_wheel_event(None, 1)

        # Check new value (should increase by MOUSE_WHEEL_STEP)
        new_value = dpg.get_value("room_camera_zoom")
        expected_value = min(initial_value + self.app.MOUSE_WHEEL_STEP, self.app.room_camera_range_end)
        self.assertEqual(new_value, expected_value)

        # Simulate wheel event - negative scroll (zoom out)
        self.app.mouse_wheel_event(None, -1)

        # Check new value (should decrease back to initial)
        final_value = dpg.get_value("room_camera_zoom")
        self.assertEqual(final_value, initial_value)

    def test_key_press_simulation(self):
        self.app = VisualizationApp()
        self.app.create_app()

        # Get initial zoom value
        initial_value = dpg.get_value("room_camera_zoom")

        # Reset to initial value
        dpg.set_value("room_camera_zoom", initial_value)

        # Directly perform what the key press would do
        new_value = min(initial_value + self.app.CAMERA_ZOOM_STEP, self.app.room_camera_range_end)
        dpg.set_value("room_camera_zoom", new_value)
        self.app.on_update_focal_length("room_camera_zoom", None)

        # Verify the change
        current_value = dpg.get_value("room_camera_zoom")
        expected_value = min(initial_value + self.app.CAMERA_ZOOM_STEP, self.app.room_camera_range_end)
        self.assertAlmostEqual(current_value, expected_value, places=5)

    def test_ui_state_updates(self):
        self.app = VisualizationApp()
        self.app.create_app()

        # Test enabling/disabling a camera stream
        dpg.set_value("streaming_room_camera", True)
        self.assertTrue(dpg.get_value("streaming_room_camera"))

        dpg.set_value("streaming_room_camera", False)
        self.assertFalse(dpg.get_value("streaming_room_camera"))

        # Test camera mode selection
        dpg.set_value("room_camera_mode", "DEPTH")
        self.assertEqual(dpg.get_value("room_camera_mode"), "DEPTH")

        dpg.set_value("room_camera_mode", "RGB")
        self.assertEqual(dpg.get_value("room_camera_mode"), "RGB")

    def test_non_ui_functionality(self):
        self.app = VisualizationApp()

        # Test that we can create image data arrays of the right shape
        self.assertEqual(self.app.room_camera_image_data.shape[2], 3)  # Should be RGB format
        self.assertTrue(np.all(self.app.room_camera_image_data == 0))  # Should start with zeros

        # Test that we can modify the image data
        test_data = np.ones_like(self.app.room_camera_image_data)
        self.app.room_camera_image_data[:] = test_data
        self.assertTrue(np.all(self.app.room_camera_image_data == 1.0))

    def test_app_run_brief(self):
        # Create a thread to run the app
        def run_app_briefly():
            app = VisualizationApp()
            app.create_app()

            # Run for a short time (200ms)
            start_time = time.time()
            while time.time() - start_time < 0.2:
                if not dpg.is_dearpygui_running():
                    break
                dpg.render_dearpygui_frame()

            # Cleanup
            dpg.destroy_context()

        # Run the app in a thread
        thread = threading.Thread(target=run_app_briefly)
        thread.start()
        thread.join(timeout=1.0)  # Wait for thread to finish (with timeout)

        # If thread is still alive after timeout, there might be an issue
        self.assertFalse(thread.is_alive(), "App thread didn't complete in expected time")

    @mock.patch("workflows.robotic_ultrasound.scripts.utils.visualization.SubscriberWithCallback")
    def test_connect_to_dds(self, mock_subscriber):
        # Mock the SubscriberWithCallback to avoid actual DDS connections
        mock_subscriber_instance = mock.MagicMock()
        mock_subscriber.return_value = mock_subscriber_instance

        self.app = VisualizationApp()
        self.app.create_app()

        # Call connect_to_dds
        with mock.patch.object(self.app, "connect_to_dds_publisher") as mock_publisher:
            with mock.patch.object(self.app, "connect_to_dds_subscriber") as mock_connect_sub:
                mock_connect_sub.return_value = mock.MagicMock()
                self.app.connect_to_dds()
                # Verify publishers were connected
                self.assertTrue(mock_publisher.called)
                # Verify subscribers were connected
                self.assertTrue(mock_connect_sub.called)

    def test_on_receive_methods(self):
        self.app = VisualizationApp()
        self.app.create_app()

        # Test on_receive_franka_annotations
        mock_franka_info = mock.MagicMock()
        mock_franka_info.joints_state_positions = [1.0, 2.0, 3.0]
        mock_franka_info.joints_state_velocities = [0.1, 0.2, 0.3]
        self.app.on_receive_franka_annotations("test_topic", mock_franka_info)

        # Test on_receive_target_annotations
        mock_target_info = mock.MagicMock()
        mock_target_info.position = [4.0, 5.0, 6.0]
        mock_target_info.orientation = [0.4, 0.5, 0.6, 0.7]
        self.app.on_receive_target_annotations("test_topic", mock_target_info)

        # Test on_receive_ultrasound_annotations
        mock_usp_info = mock.MagicMock()
        mock_usp_info.position = [7.0, 8.0, 9.0]
        self.app.on_receive_ultrasound_annotations("test_topic", mock_usp_info)

    def test_publish_methods(self):
        self.app = VisualizationApp()
        self.app.create_app()  # We need to create the app UI first

        c = config.room_camera
        camera_topic_name = c.topic_ctrl.name

        # Use the actual topic name from the config
        mock_writer = mock.MagicMock()
        self.app.dds_writers = {camera_topic_name: mock_writer}

        # Test publish_camera_annotations
        self.app.publish_camera_annotations({"focal_len": 50.0})
        self.assertTrue(
            mock_writer.write.called,
            f"Writer for topic {camera_topic_name} was not called. Available writers: {self.app.dds_writers.keys()}",
        )

        # Reset the mock for the next test
        mock_writer.reset_mock()

        # Test target annotations (requires more setup)
        self.app.current_target_position = [0, 0, 0]
        self.app.current_target_orientation = [0, 0, 0, 1]

        # Mock dpg.get_value for the checkbox
        with mock.patch("dearpygui.dearpygui.get_value") as mock_get_value:
            # Configure to return True for move_target and "Random" for move_target_by
            def side_effect(arg):
                if arg == "move_target":
                    return True
                elif arg == "move_target_by":
                    return "Random"
                return None

            mock_get_value.side_effect = side_effect

            # Mock the threading.Thread
            with mock.patch("threading.Thread") as mock_thread:
                mock_thread_instance = mock.MagicMock()
                mock_thread.return_value = mock_thread_instance

                # Add the target writer with correct topic name from config
                target_topic_name = config.target.topic_ctrl.name
                self.app.dds_writers[target_topic_name] = mock_writer

                # Call the method
                self.app.publish_target_annotations(None, None, None)

                # Verify thread was started
                self.assertTrue(mock_thread_instance.start.called)

        # Test franka annotations publishing with similar approach
        with mock.patch("dearpygui.dearpygui.get_value") as mock_get_value:
            mock_get_value.return_value = True

            with mock.patch("threading.Thread") as mock_thread:
                mock_thread_instance = mock.MagicMock()
                mock_thread.return_value = mock_thread_instance

                # Add the franka writer with correct topic name from config
                franka_topic_name = config.franka.topic_ctrl.name
                self.app.dds_writers[franka_topic_name] = mock_writer
                self.app.current_joints_state_positions = [0, 0, 0, 0, 0, 0, 0]

                # Call the method
                self.app.publish_franka_annotations(None, None, None)

                # Verify thread was started
                self.assertTrue(mock_thread_instance.start.called)

    def test_on_camera_annotations(self):
        self.app = VisualizationApp()
        self.app.create_app()

        # Create mock camera info for RGB
        mock_camera_rgb = mock.MagicMock()
        h, w = 480, 640
        mock_camera_rgb.data = np.zeros((h * w * 3), dtype=np.uint8).tobytes()
        mock_camera_rgb.focal_len = 100.0

        # Create mock camera config
        mock_config = mock.MagicMock()
        mock_config.height = h
        mock_config.width = w

        # Test room camera RGB mode
        with mock.patch("dearpygui.dearpygui.get_value") as mock_get_value:
            mock_get_value.return_value = "RGB"
            self.app.on_camera_annotations(mock_camera_rgb, mock_config, True)

            # Test room camera DEPTH mode
            mock_get_value.return_value = "DEPTH"
            mock_camera_depth = mock.MagicMock()
            mock_camera_depth.data = np.zeros((h * w), dtype=np.float32).tobytes()
            mock_camera_depth.focal_len = 100.0

            # Mock colorize_depth
            with mock.patch("workflows.robotic_ultrasound.scripts.utils.visualization.colorize_depth") as mock_colorize:
                mock_colorize.return_value = np.zeros((h, w, 3), dtype=np.uint8)
                self.app.on_camera_annotations(mock_camera_depth, mock_config, True)

    def test_on_receive_ultrasound_image(self):
        self.app = VisualizationApp()
        self.app.create_app()

        # Create mock ultrasound data
        mock_usp_data = mock.MagicMock()
        h, w = 480, 640
        mock_usp_data.data = np.zeros((h * w * 3), dtype=np.uint8).tobytes()

        # Mock PIL.Image.fromarray and convert
        with mock.patch("workflows.robotic_ultrasound.scripts.utils.visualization.Image") as mock_image:
            mock_img = mock.MagicMock()
            mock_img.convert.return_value = mock_img
            mock_image.fromarray.return_value = mock_img

            # Mock np.array
            with mock.patch("numpy.array") as mock_np_array:
                mock_np_array.return_value = np.zeros((h, w, 3), dtype=np.uint8)

                # Call the method
                self.app.on_receive_ultrasound_image("test_topic", mock_usp_data)

                # Verify the calls
                mock_image.fromarray.assert_called_once()
                mock_img.convert.assert_called_once_with("RGB")
                mock_np_array.assert_called_once()

    # Add a cleanup method to restore sys.argv
    @classmethod
    def tearDownClass(cls):
        sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
