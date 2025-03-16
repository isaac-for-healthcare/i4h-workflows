# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
