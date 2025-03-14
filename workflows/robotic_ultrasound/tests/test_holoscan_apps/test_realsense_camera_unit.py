import unittest

from holoscan_apps.realsense.camera import RealsenseApp


class TestRealSenseCameraUnit(unittest.TestCase):
    def test_realsense_app(self):
        domain_id = 4321
        height, width = 480, 640
        topic_rgb = "topic_test_camera_rgb"
        topic_depth = "topic_test_camera_depth"
        device_idx = 0
        framerate = 30
        test = False
        count = 10

        app = RealsenseApp(
            domain_id,
            height,
            width,
            topic_rgb,
            topic_depth,
            device_idx,
            framerate,
            test,
            count,
        )

        self.assertIsNotNone(app)
        self.assertEqual(app.width, width)
        self.assertEqual(app.height, height)
        self.assertEqual(app.framerate, framerate)
        self.assertEqual(app.topic_rgb, topic_rgb)
        self.assertEqual(app.topic_depth, topic_depth)
