import asyncio
import queue
import unittest
from unittest import skipUnless
from unittest.mock import Mock, patch

# Try importing rti.connextdds
try:
    import rti.connextdds as dds
    RTI_AVAILABLE = True
except ImportError:
    RTI_AVAILABLE = False

from ..subscriber import Subscriber, SubscriberWithCallback, SubscriberWithQueue


class MockSubscriber(Subscriber):
    """Concrete implementation of Subscriber for testing"""
    def consume(self, data) -> None:
        if isinstance(data, list):
            data.append("test")
        elif isinstance(data, str):
            data += "test"


@skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed")
class TestSubscriber(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock the DDS components
        self.mock_participant = Mock(spec=dds.DomainParticipant)
        self.mock_topic = Mock(spec=dds.Topic)
        self.mock_reader = Mock(spec=dds.DataReader)

        # Create patches for DDS components
        self.participant_patcher = patch('rti.connextdds.DomainParticipant', return_value=self.mock_participant)
        self.topic_patcher = patch('rti.connextdds.Topic', return_value=self.mock_topic)
        self.reader_patcher = patch('rti.connextdds.DataReader', return_value=self.mock_reader)

        # Start the patches
        self.participant_patcher.start()
        self.topic_patcher.start()
        self.reader_patcher.start()

        # Create test instance
        self.subscriber = MockSubscriber(
            topic="test_topic",
            cls=str,
            period=0.1,
            domain_id=0
        )

    def tearDown(self):
        """Clean up after each test method"""
        self.participant_patcher.stop()
        self.topic_patcher.stop()
        self.reader_patcher.stop()
        if self.subscriber:
            self.subscriber.stop()

    def test_init(self):
        """Test subscriber initialization"""
        self.assertEqual(self.subscriber.topic, "test_topic")
        self.assertEqual(self.subscriber.cls, str)
        self.assertEqual(self.subscriber.period, 0.1)
        self.assertEqual(self.subscriber.domain_id, 0)
        self.assertTrue(self.subscriber.add_to_queue)
        self.assertIsInstance(self.subscriber.data_q, queue.Queue)
        self.assertIsNone(self.subscriber.dds_reader)
        self.assertIsNone(self.subscriber.stop_event)

    def test_start_stop(self):
        """Test start and stop functionality"""
        # Test start
        self.subscriber.start()
        self.assertIsNotNone(self.subscriber.stop_event)
        self.assertIsNotNone(self.subscriber.dds_reader)

        # Test stop
        self.subscriber.stop()
        self.assertIsNone(self.subscriber.stop_event)

    def test_read_data(self):
        """Test reading data from queue"""
        test_data = "test_data"
        self.subscriber.data_q.put(test_data)

        # Test reading data
        data = self.subscriber.read_data()
        self.assertEqual(data, test_data)

        # Test empty queue
        data = self.subscriber.read_data()
        self.assertIsNone(data)

    def test_read(self):
        """Test processing all data in queue"""
        test_data = ["data1", "data2", "data3"]
        for data in test_data:
            self.subscriber.data_q.put(data)

        exec_time = self.subscriber.read(0.1, 1.0)
        self.assertGreaterEqual(exec_time, 0)
        self.assertTrue(self.subscriber.data_q.empty())

    def run_async_test(self, coro):
        """Helper function to run async tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_async_reading(self):
        """Wrapper for async test"""
        self.run_async_test(self.async_test_impl())

    async def async_test_impl(self):
        """Actual async test implementation"""
        test_data = ["data1", "data2", "data3"]
        async def mock_data_source():
            for data in test_data:
                await asyncio.sleep(0.1)
                yield data
        self.subscriber.dds_reader = self.mock_reader
        self.mock_reader.topic_name = "test_topic"
        self.mock_reader.take_data_async = mock_data_source
        await self.subscriber.read_async()
        self.assertEqual(self.subscriber.data_q.get(), test_data[0])
        self.assertEqual(self.subscriber.data_q.qsize(), len(test_data) - 1)


class TestSubscriberWithQueue(unittest.TestCase):
    def setUp(self):
        self.subscriber = SubscriberWithQueue(
            domain_id=0,
            topic="test_topic",
            cls=str,
            period=0.1
        )

    def test_consume_raises_error(self):
        """Test that consume() logs an error"""
        with self.assertLogs(level='ERROR') as log:
            self.subscriber.consume("test_data")
            self.assertIn("This should not happen", log.output[0])


class TestSubscriberWithCallback(unittest.TestCase):
    def setUp(self):
        self.callback_called = False
        self.callback_data = None

        def test_callback(topic, data):
            self.callback_called = True
            self.callback_data = data + "test"

        self.subscriber = SubscriberWithCallback(
            cb=test_callback,
            domain_id=0,
            topic="test_topic",
            cls=str,
            period=0.1
        )

    def test_callback(self):
        """Test that callback is properly called"""
        test_data = "test_data"
        self.subscriber.consume(test_data)

        self.assertTrue(self.callback_called)
        self.assertEqual(self.callback_data, f"{test_data}test")


if __name__ == '__main__':
    unittest.main()
