import time
import unittest
from unittest.mock import MagicMock, patch

from parameterized import parameterized
from robotic_ultrasound.scripts.simulation.utils.ratelimit import (
    I4H_SIMULATION_PHYX_CALLBACKS,
    RateLimitedCallback,
    RateStats,
    add_physx_step_callback,
    remove_physx_callback,
    remove_physx_callbacks,
)

TEST_CASES = [
    ("slow_execution", 15, 3.0, 3.0),  # Simulating slow execution
    ("fast_execution", 45, 3.0, 3.0),  # Simulating fast execution 
    ("very_slow", 5, 3.0, 3.0),  # Simulating very slow execution
]

class TestRateLimitedCallback(unittest.TestCase):
    def setUp(self):
        self.mock_world = MagicMock()
        self.mock_world.current_time = 0.0
        
        # Mock callback function
        self.callback_called = 0
        def test_callback(period, current_time):
            self.callback_called += 1
            return time.time()
        self.test_callback = test_callback

    def tearDown(self):
        I4H_SIMULATION_PHYX_CALLBACKS.clear()

    def test_init_with_invalid_period(self):
        with self.assertRaises(ValueError):
            RateLimitedCallback("test", 0, self.test_callback, self.mock_world)

    def test_get_current_time(self):
        callback = RateLimitedCallback("test", 10, self.test_callback, self.mock_world)
        self.assertEqual(callback.get_current_time(), 0.0)

        # Test fallback to system time
        delattr(self.mock_world, 'current_time')
        with patch('time.time', return_value=123.0):
            self.assertEqual(callback.get_current_time(), 123.0)

    def test_rate_limiting(self):
        self.mock_world.current_time = 0.0
        target_period = 10  # 10 Hz
        callback = RateLimitedCallback(
            "test",
            target_period,
            self.test_callback,
            self.mock_world,
            adaptive_period=False
        )

        # Simulate multiple physics steps
        for i in range(100):
            self.mock_world.current_time = i * 0.01  # 0.01s steps
            callback.rate_limit(0.01)

        # Check if execution period is approximately correct
        expected_calls = (100 * 0.01) * (1 / target_period)
        delta = 2
        self.assertAlmostEqual(
            self.callback_called,
            expected_calls,
            delta=delta
        )

    @parameterized.expand(TEST_CASES)
    def test_adaptive_rate_adjustment(self, name, exec_count, real_time, interval_time):
        callback = RateLimitedCallback(
            "test",
            1 / 10,
            self.test_callback,
            self.mock_world,
            adaptive_period=True
        )

        # Simulate execution with different speeds
        with patch('time.time', return_value=real_time):
            callback.stats.exec_count = exec_count
            callback.update_period_stats(real_time, interval_time)

        # Check if period was adjusted    
        self.assertNotEqual(callback.adj_period, callback.period)
        
        expected_hz = 10
        
        if exec_count < expected_hz * interval_time:
            self.assertLess(callback.adj_period, callback.period)
        elif exec_count > expected_hz * interval_time:
            self.assertGreater(callback.adj_period, callback.period)

    def test_get_stats(self):
        callback = RateLimitedCallback(
            "test",
            1 / 10,
            self.test_callback,
            self.mock_world
        )
        stats = callback.get_stats()
        self.assertIsInstance(stats, RateStats)
        self.assertEqual(stats.target_period, 1 / 10)


class TestPhysXCallbacks(unittest.TestCase):
    def setUp(self):
        self.mock_world = MagicMock()
        self.test_callback = MagicMock()
        I4H_SIMULATION_PHYX_CALLBACKS.clear()

    def test_add_physx_step_callback(self):
        add_physx_step_callback("test", 10, self.test_callback, self.mock_world)
        
        self.assertIn("test", I4H_SIMULATION_PHYX_CALLBACKS)
        self.mock_world.add_physics_callback.assert_called_once()

    def test_remove_physx_callback(self):
        # First add a callback
        add_physx_step_callback("test", 10, self.test_callback, self.mock_world)
        
        # Then remove it
        remove_physx_callback("test", self.mock_world)
        
        self.assertNotIn("test", I4H_SIMULATION_PHYX_CALLBACKS)
        self.mock_world.remove_physics_callback.assert_called_once_with("test")


    def test_remove_physx_callbacks(self):
        # Add multiple callbacks
        add_physx_step_callback("test1", 10, self.test_callback, self.mock_world)
        add_physx_step_callback("test2", 20, self.test_callback, self.mock_world)
        
        self.assertEqual(len(I4H_SIMULATION_PHYX_CALLBACKS), 2)
        # Remove all callbacks
        remove_physx_callbacks(self.mock_world)
        
        self.assertEqual(len(I4H_SIMULATION_PHYX_CALLBACKS), 0)
        self.assertEqual(
            self.mock_world.remove_physics_callback.call_count,
            2
        )


if __name__ == '__main__':
    unittest.main()
