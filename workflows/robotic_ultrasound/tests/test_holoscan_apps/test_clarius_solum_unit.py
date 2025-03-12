import unittest

from holoscan_apps.clarius_solum.clarius_solum import ClariusSolumApp


class TestClariusSolumUnit(unittest.TestCase):
    def test_clarius_solum_app(self):

        ip = "192.168.1.1"
        port = 5858
        cert = "/path/to/cert"
        model = "C3HD3"
        application = "abdomen"
        domain_id = 421
        height = 480
        width = 640
        topic_out = "topic_ultrasound_stream"
        test = False

        app = ClariusSolumApp(ip, port, cert, model, application, domain_id, height, width, topic_out, test)

        self.assertIsNotNone(app)
        self.assertEqual(app.ip, ip)
        self.assertEqual(app.port, port)
        self.assertEqual(app.cert, cert)
        self.assertEqual(app.model, model)
        self.assertEqual(app.app, application)
        self.assertEqual(app.domain_id, domain_id)
        self.assertEqual(app.height, height)
        self.assertEqual(app.width, width)
        self.assertEqual(app.topic_out, topic_out)
        self.assertEqual(app.show_holoviz, test)
