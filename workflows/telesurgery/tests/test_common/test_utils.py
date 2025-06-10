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

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Mock ntplib module before importing utils to handle missing dependency
sys.modules["ntplib"] = MagicMock()


class TestGetNtpOffset(unittest.TestCase):
    """Test cases for get_ntp_offset function."""

    @patch("common.utils.ntplib.NTPClient")
    def test_get_ntp_offset_success(self, mock_ntp_client_class):
        """Test successful NTP offset retrieval."""
        from common.utils import get_ntp_offset

        # Mock the NTP client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.offset = 0.123456
        mock_client.request.return_value = mock_response
        mock_ntp_client_class.return_value = mock_client

        # Test with explicit parameters
        offset = get_ntp_offset(ntp_server="pool.ntp.org", ntp_port=123, version=3)

        self.assertEqual(offset, 0.123456)
        mock_ntp_client_class.assert_called_once()
        mock_client.request.assert_called_once_with("pool.ntp.org", port=123, version=3)

    @patch("common.utils.ntplib.NTPClient")
    def test_get_ntp_offset_exception(self, mock_ntp_client_class):
        """Test NTP offset when request raises an exception."""
        from common.utils import get_ntp_offset

        # Mock the NTP client to raise an exception
        mock_client = Mock()
        mock_client.request.side_effect = Exception("Connection timeout")
        mock_ntp_client_class.return_value = mock_client

        # Test that exception is handled and returns 0
        offset = get_ntp_offset(ntp_server="pool.ntp.org", ntp_port=123, version=3)

        self.assertEqual(offset, 0)
        mock_ntp_client_class.assert_called_once()
        mock_client.request.assert_called_once_with("pool.ntp.org", port=123, version=3)

    def test_get_ntp_offset_no_server(self):
        """Test NTP offset when no server is provided."""
        from common.utils import get_ntp_offset

        # Test with None server
        offset = get_ntp_offset(ntp_server=None)
        self.assertEqual(offset, 0)

        # Test with empty string server
        offset = get_ntp_offset(ntp_server="")
        self.assertEqual(offset, 0)

    @patch.dict(os.environ, {"NTP_SERVER_HOST": "test.ntp.org", "NTP_SERVER_PORT": "1234"})
    def test_get_ntp_offset_env_variables(self):
        """Test NTP offset using environment variables for configuration."""
        # Need to reload the module after setting environment variables
        import importlib

        import common.utils

        importlib.reload(common.utils)
        from common.utils import get_ntp_offset

        # Mock the NTP client and response after reload
        with patch("common.utils.ntplib.NTPClient") as mock_ntp_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.offset = -0.987654
            mock_client.request.return_value = mock_response
            mock_ntp_client_class.return_value = mock_client

            # Test with default parameters (should use env vars)
            offset = get_ntp_offset()

            self.assertEqual(offset, -0.987654)
            mock_client.request.assert_called_once_with("test.ntp.org", port=1234, version=3)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_ntp_offset_no_env_variables(self):
        """Test NTP offset when no environment variables are set."""
        # Need to reload the module after clearing environment variables
        import importlib

        import common.utils

        importlib.reload(common.utils)
        from common.utils import get_ntp_offset

        # Test with no environment variables set
        offset = get_ntp_offset()
        self.assertEqual(offset, 0)

    @patch.dict(os.environ, {"NTP_SERVER_HOST": "test.ntp.org"}, clear=True)
    def test_get_ntp_offset_default_port(self):
        """Test NTP offset with default port when NTP_SERVER_PORT is not set."""
        # Need to reload the module after setting environment variables
        import importlib

        import common.utils

        importlib.reload(common.utils)
        from common.utils import get_ntp_offset

        # Mock the NTP client and response after reload
        with patch("common.utils.ntplib.NTPClient") as mock_ntp_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.offset = 0.555555
            mock_client.request.return_value = mock_response
            mock_ntp_client_class.return_value = mock_client

            # Test with default port (should be 123)
            offset = get_ntp_offset()

            self.assertEqual(offset, 0.555555)
            mock_client.request.assert_called_once_with("test.ntp.org", port=123, version=3)


if __name__ == "__main__":
    unittest.main()
