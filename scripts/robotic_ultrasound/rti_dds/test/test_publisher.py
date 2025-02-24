import unittest
from unittest import skipUnless
from unittest.mock import Mock, patch

# Try importing rti.connextdds
try:
    import rti.connextdds as dds
    RTI_AVAILABLE = True
except ImportError:
    RTI_AVAILABLE = False

from ..publisher import Publisher


# Add skipUnless decorator to the test class
@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed")
class TestPublisher(unittest.TestCase):

    class MockPublisher(Publisher):
        """Concrete implementation of Publisher for testing"""
        def produce(self, dt, sim_time):
            return "test_data"

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock the DDS components
        self.mock_participant = Mock(spec=dds.DomainParticipant)
        self.mock_topic = Mock(spec=dds.Topic)
        self.mock_writer = Mock(spec=dds.DataWriter)

        # Create patch for DDS components
        self.participant_patcher = patch('rti.connextdds.DomainParticipant', return_value=self.mock_participant)
        self.topic_patcher = patch('rti.connextdds.Topic', return_value=self.mock_topic)
        self.writer_patcher = patch('rti.connextdds.DataWriter', return_value=self.mock_writer)

        # Start the patches
        self.participant_patcher.start()
        self.topic_patcher.start()
        self.writer_patcher.start()

        # Create test instance
        self.publisher = self.MockPublisher(
            topic="test_topic",
            cls=str,
            period=0.1,
            domain_id=0
        )

    def tearDown(self):
        """Clean up after each test method"""
        self.participant_patcher.stop()
        self.topic_patcher.stop()
        self.writer_patcher.stop()

    def test_init(self):
        """Test publisher initialization"""
        self.assertEqual(self.publisher.topic, "test_topic")
        self.assertEqual(self.publisher.cls, str)
        self.assertEqual(self.publisher.period, 0.1)
        self.assertEqual(self.publisher.domain_id, 0)
        self.assertIsNotNone(self.publisher.logger)
        self.assertIsNotNone(self.publisher.dds_writer)

    def test_write(self):
        """Test write method"""
        # Test successful write
        exec_time = self.publisher.write(0.1, 1.0)
        self.mock_writer.write.assert_called_once_with("test_data")
        self.assertGreaterEqual(exec_time, 0)

        # Test write with no writer
        self.publisher.dds_writer = None
        exec_time = self.publisher.write(0.1, 1.0)
        self.assertEqual(exec_time, 0)

    def test_produce_abstract(self):
        """Test that produce() is abstract and must be implemented"""
        with self.assertRaises(TypeError):
            Publisher("test", str, 0.1, 0)


if __name__ == '__main__':
    unittest.main()
