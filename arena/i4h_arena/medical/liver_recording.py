# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete numerical continuation inside Isaac Lab's initial-state dataset."""

import numpy as np
import torch
from isaaclab.managers import RecorderTerm


def record_bundle(bundle, count):
    if isinstance(bundle, dict):
        return {key: record_bundle(value, count) for key, value in bundle.items()}
    value = torch.from_numpy(np.asarray(bundle).copy())
    return value.unsqueeze(0).expand(count, *value.shape)


def replay_bundle(bundle):
    if isinstance(bundle, dict):
        return {key: replay_bundle(value) for key, value in bundle.items()}
    # EpisodeData retains a leading one-sample time dimension for initial_state.
    return bundle[0].detach().cpu().numpy().copy()


class LiverInitialStateRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        count = self._env.num_envs if env_ids is None else len(env_ids)
        return "initial_state/i4h", record_bundle(self._env.capture_complete_state(), count)
