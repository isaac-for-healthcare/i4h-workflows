import time
from typing import Callable


class RateLimitedCallback:
    def __init__(
        self,
        name: str,
        rate: float,
        fn: Callable,
        world,
        start_time: float = 0.0,
        adaptive_rate: bool = True,
    ) -> None:
        self.name = name
        self.fn = fn  # function to call at rate
        self.world = world
        self.rate = rate  # 1/Hz
        self.previous_step_time = 0.0
        self.accumulated_time = 0.0
        self.last_exec_time = 0.0

        self.adaptive_rate = adaptive_rate
        self.start_time = start_time  # real world time at which the simulation started
        self.interval = 3.0  # seconds
        self.accumulated_interval_time = 0.0
        self.exec_count = 0
        self.actual_rate = rate
        self.adj_rate = rate
        self.rates_diff = 0.0

    def rate_limit(self, dt) -> None:
        real_time = time.time() - self.start_time
        interval_time = real_time - self.accumulated_interval_time

        # Sample the actual rate each interval, by counting executions
        # Find the difference and set new adjusted rate
        if interval_time >= self.interval and self.adaptive_rate:
            self.accumulated_interval_time = real_time
            interval_rate = self.exec_count / interval_time
            self.actual_rate = (1 / interval_rate) if interval_rate > 0 else self.rate
            self.exec_count = 0

            # Adjust rate
            self.rates_diff = self.rate - self.actual_rate
            if abs(self.rate - self.actual_rate) > 0.001:
                self.adj_rate += self.rates_diff

        if not self.adaptive_rate:
            self.adj_rate = self.rate

        # -> Times here are simulated physical times
        current_time = self.world.current_time if hasattr(self.world, "current_time") else time.time()
        elapsed_time = current_time - self.previous_step_time
        self.previous_step_time = current_time
        self.accumulated_time += elapsed_time

        if self.accumulated_time >= self.adj_rate:
            self.last_exec_time = self.fn(self.rate, current_time)
            self.accumulated_time -= self.adj_rate
            self.exec_count += 1


PHYX_CALLBACKS = {}


def add_physx_step_callback(name: str, hz: float, fn: Callable, world) -> None:
    rate_limited_callback = RateLimitedCallback(name, hz, fn, world)
    if hasattr(world, "add_physics_callback") and callable(world.add_physics_callback):
        world.add_physics_callback(name, rate_limited_callback.rate_limit)
    else:
        print(f"Something went wrong adding physics callback for {name}")
    PHYX_CALLBACKS[name] = rate_limited_callback
    return


def remove_physx_callback(name: str, world) -> None:
    cb = PHYX_CALLBACKS.pop(name, None) if name else None
    if cb is not None:
        if hasattr(world, "remove_physics_callback") and callable(world.remove_physics_callback):
            world.remove_physics_callback(name)
        else:
            print(f"Something went wrong removing physics callback for {name}")


def remove_physx_callbacks(world) -> None:
    for name, cb in PHYX_CALLBACKS.items():
        if hasattr(world, "remove_physics_callback") and callable(world.remove_physics_callback):
            world.remove_physics_callback(name)
        else:
            print(f"Something went wrong removing physics callback for {name}")
        del cb
