import time
from dataclasses import dataclass
from typing import Any, Callable
from warnings import warn


@dataclass
class RateStats:
    """Statistics about the rate limiting execution."""
    actual_rate: float  # Measured execution rate in Hz
    target_rate: float  # Target rate in Hz
    exec_count: int     # Number of executions in current interval
    last_exec_time: float  # Last execution timestamp


class RateLimitedCallback:
    """A rate-limited callback wrapper that ensures functions are called at a specified frequency.
    
    This class implements rate limiting with optional adaptive rate adjustment to maintain
    the desired callback frequency even under varying system load conditions.
    
    Args:
        name: Identifier for the callback
        rate: Period between executions in seconds (1/Hz)
        fn: The callback function to be rate-limited
        world: The simulation world object that provides timing information
        start_time: Real-world timestamp when the simulation started
        adaptive_rate: Whether to dynamically adjust timing to maintain target rate
        
    """

    def __init__(
        self,
        name: str,
        rate: float,
        fn: Callable[[float, float], float],
        world: Any,
        start_time: float = 0.0,
        adaptive_rate: bool = True,
    ) -> None:
        if rate <= 0:
            raise ValueError("Rate must be positive")
            
        self.name = name
        self.fn = fn
        self.world = world
        self.rate = rate
        self.adaptive_rate = adaptive_rate
        self.start_time = start_time
        
        # Timing control
        self.previous_step_time: float = 0.0
        self.accumulated_time: float = 0.0
        
        # Adaptive rate control
        self.interval: float = 3.0  # seconds
        self.accumulated_interval_time: float = 0.0
        self.stats = RateStats(
            actual_rate=rate,
            target_rate=rate,
            exec_count=0,
            last_exec_time=0.0
        )
        self.adj_rate: float = self.rate
        self._max_rate_diff: float = 0.1  # Maximum allowed rate adjustment per interval

    def get_current_time(self) -> float:
        """Get current time from world if available, otherwise use system time."""
        return (
            self.world.current_time 
            if hasattr(self.world, "current_time") 
            else time.time()
        )

    def update_rate_stats(self, real_time: float, interval_time: float) -> None:
        """Update execution rate statistics and adjust rate if needed."""
        if not self.adaptive_rate or interval_time < self.interval:
            return

        self.accumulated_interval_time = real_time
        interval_rate = self.stats.exec_count / interval_time
        self.stats.actual_rate = (1 / interval_rate) if interval_rate > 0 else self.rate
        self.stats.exec_count = 0

        rate_diff = self.rate - self.stats.actual_rate
        if abs(rate_diff) > 0.001:
            self.adj_rate += rate_diff

    def rate_limit(self, dt: float) -> None:
        """Execute the callback function if enough time has elapsed.
        
        Args:
            dt: Time delta since last physics step
        """
        real_time = time.time() - self.start_time
        interval_time = real_time - self.accumulated_interval_time
        
        # Update rate statistics and adjust if needed
        self.update_rate_stats(real_time, interval_time)

        if not self.adaptive_rate:
            self.adj_rate = self.rate

        # Handle timing for callback execution
        current_time = self.get_current_time()
        elapsed_time = current_time - self.previous_step_time
        self.previous_step_time = current_time
        self.accumulated_time += elapsed_time

        if self.accumulated_time >= self.adj_rate:
            try:
                self.stats.last_exec_time = self.fn(self.rate, current_time)
                self.accumulated_time -= self.adj_rate
                self.stats.exec_count += 1
            except Exception as e:
                warn(f"Error in callback {self.name}: {str(e)}")

    def get_stats(self) -> RateStats:
        """Return current rate limiting statistics."""
        return self.stats

I4H_SIMULATION_PHYX_CALLBACKS = {}


def add_physx_step_callback(name: str, hz: float, fn: Callable, world: Any) -> None:
    """Register a rate-limited callback to be executed during physics simulation steps.
    
    Args:
        name: Unique identifier for the callback
        hz: Target execution frequency in Hertz
        fn: Callback function to be executed. Should accept (rate: float, current_time: float)
        world: Simulation world object that implements add_physics_callback method
    
    Note:
        The callback will be stored in PHYX_CALLBACKS dictionary and registered with
        the physics engine through world.add_physics_callback.
    """
    rate_limited_callback = RateLimitedCallback(name, hz, fn, world)
    if hasattr(world, "add_physics_callback") and callable(world.add_physics_callback):
        world.add_physics_callback(name, rate_limited_callback.rate_limit)
    else:
        warn(f"Something went wrong adding physics callback for {name}")
    I4H_SIMULATION_PHYX_CALLBACKS[name] = rate_limited_callback
    return


def remove_physx_callback(name: str, world: Any) -> None:
    """Remove a specific physics callback from the simulation.
    
    Args:
        name: Identifier of the callback to remove
        world: Simulation world object that implements remove_physics_callback method
        
    Note:
        Removes the callback from both I4H_SIMULATION_PHYX_CALLBACKS dictionary and
        the physics engine through world.remove_physics_callback.
    """
    cb = I4H_SIMULATION_PHYX_CALLBACKS.pop(name, None) if name else None
    if cb is not None:
        if hasattr(world, "remove_physics_callback") and callable(world.remove_physics_callback):
            world.remove_physics_callback(name)
        else:
            warn(f"Something went wrong removing physics callback for {name}")


def remove_physx_callbacks(world: Any) -> None:
    """Remove all registered physics callbacks from the simulation.
    
    Args:
        world: Simulation world object that implements remove_physics_callback method
        
    Note:
        Clears all callbacks from I4H_SIMULATION_PHYX_CALLBACKS dictionary and
        removes them from the physics engine.
    """
    names = list(I4H_SIMULATION_PHYX_CALLBACKS.keys())
    for name in names:
        if hasattr(world, "remove_physics_callback") and callable(world.remove_physics_callback):
            world.remove_physics_callback(name)
        else:
            warn(f"Something went wrong removing physics callback for {name}")
        if I4H_SIMULATION_PHYX_CALLBACKS.pop(name, None) is None:
            warn(f"Could not find registered physics callback with name={name}.")
