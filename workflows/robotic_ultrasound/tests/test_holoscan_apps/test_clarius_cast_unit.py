import unittest

from holoscan_apps.clarius_cast.clarius_cast import ClariusCastApp


class TestClariusCastUnit(unittest.TestCase):
    def test_clarius_cast_app(self):
        ip = "192.168.1.1"
        port = 5858
        domain_id = 421
        height = 480
        width = 640
        topic_out = "topic_ultrasound_stream"
        test = False

        app = ClariusCastApp(ip, port, domain_id, height, width, topic_out, test)

        self.assertIsNotNone(app)
        self.assertEqual(app.ip, ip)
        self.assertEqual(app.port, port)
        self.assertEqual(app.domain_id, domain_id)
        self.assertEqual(app.height, height)
        self.assertEqual(app.width, width)
        self.assertEqual(app.topic_out, topic_out)
        self.assertEqual(app.show_holoviz, test)
