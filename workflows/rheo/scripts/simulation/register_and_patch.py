# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helpers to register Workflow-specific assets and environments."""

from isaaclab_arena_environments.cli import ExampleEnvironments
from simulation.environments.g1_locomanip_observe_object_environment import ObserveObjectEnvironment
from simulation.environments.g1_locomanip_push_cart_environment import G1LocomanipPushCartEnvironment
from simulation.environments.g1_locomanip_tray_pick_and_place_environment import G1LocomanipTrayPickAndPlaceEnvironment


def register_workflow_cli():
    """Register into the global registries for CLI before the simulation app is started."""
    if G1LocomanipTrayPickAndPlaceEnvironment.name not in ExampleEnvironments:
        ExampleEnvironments.update(
            {
                G1LocomanipTrayPickAndPlaceEnvironment.name: G1LocomanipTrayPickAndPlaceEnvironment,
            }
        )
    if G1LocomanipPushCartEnvironment.name not in ExampleEnvironments:
        ExampleEnvironments.update(
            {
                G1LocomanipPushCartEnvironment.name: G1LocomanipPushCartEnvironment,
            }
        )
    if ObserveObjectEnvironment.name not in ExampleEnvironments:
        ExampleEnvironments.update(
            {
                ObserveObjectEnvironment.name: ObserveObjectEnvironment,
            }
        )


def register_workflow_assets():
    """Register workflow assets into the global registries for CLI before the simulation app is started."""
    import scripts.teleop_devices  # noqa: F401
    import simulation.assets.background_library  # noqa: F401
    import simulation.assets.object_library  # noqa: F401
    import simulation.embodiments.g1_patched  # noqa: F401
