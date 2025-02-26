import os
import time
import unittest
from unittest import skipUnless

from scripts.rti_dds.publisher import Publisher

try:
    import rti.connextdds as dds
    import rti.idl as idl
    license_path = os.getenv('RTI_LICENSE_FILE')
    RTI_AVAILABLE = bool(license_path and os.path.exists(license_path))
except ImportError:
    RTI_AVAILABLE = False


@idl.struct
class _TestData:
    value: int = 0
    message: str = ""


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")
class TestPublisher(unittest.TestCase):
    class TestDataPublisher(Publisher):
        """Concrete implementation of Publisher for testing"""
        def produce(self, dt: float, sim_time: float) -> _TestData:
            return _TestData(value=42, message="test")

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.domain_id = 100  # Use a unique domain ID for testing
        self.publisher = self.TestDataPublisher(
            topic="TestTopic",
            cls=_TestData,
            period=0.1,
            domain_id=self.domain_id
        )

        # Create a subscriber to verify published messages
        self.participant = dds.DomainParticipant(self.domain_id)
        self.topic = dds.Topic(self.participant, "TestTopic", _TestData)
        self.reader = dds.DataReader(
            self.participant.implicit_subscriber,
            self.topic
        )
        time.sleep(1.0)

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, 'publisher'):
            del self.publisher
        if hasattr(self, 'reader'):
            del self.reader
        if hasattr(self, 'topic'):
            del self.topic
        if hasattr(self, 'participant'):
            del self.participant
        time.sleep(0.1)  # Allow time for cleanup

    def test_init(self):
        """Test publisher initialization"""
        self.assertEqual(self.publisher.topic, "TestTopic")
        self.assertEqual(self.publisher.cls, _TestData)
        self.assertEqual(self.publisher.period, 0.1)
        self.assertEqual(self.publisher.domain_id, self.domain_id)
        self.assertIsNotNone(self.publisher.logger)
        self.assertIsNotNone(self.publisher.dds_writer)

    def test_write(self):
        """Test write method with actual DDS communication"""
        # Write data
        exec_time = self.publisher.write(0.1, 1.0)
        self.assertGreaterEqual(exec_time, 0)

        max_retries = 5
        for _ in range(max_retries):
            samples = self.reader.take()
            if samples:
                break
            time.sleep(0.2)
        self.assertEqual(len(samples), 1)
        sample_data = samples[0].data
        self.assertEqual(sample_data.value, 42)
        self.assertEqual(sample_data.message, "test")

    def test_produce_abstract(self):
        """Test that produce() is abstract and must be implemented"""
        with self.assertRaises(TypeError):
            Publisher("test", str, 0.1, 0)

if __name__ == '__main__':
    unittest.main()
