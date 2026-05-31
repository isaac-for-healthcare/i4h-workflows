# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from common.policy_stack import train_cli_main
from policy.registry import BACKENDS


def main() -> None:
    train_cli_main(BACKENDS)


if __name__ == "__main__":
    main()
