#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Git reads the BuildKit secret only when GitHub requests authentication.
set -eu
if [ ! -s /run/secrets/github_token ]; then
    echo 'Private GitHub checkout requires --secret id=github_token,env=GH_TOKEN' >&2
    exit 1
fi
case "${1:-}" in
    "Username for 'https://github.com':"*) printf '%s\n' 'x-access-token' ;;
    "Password for 'https://x-access-token@github.com':"*) cat /run/secrets/github_token ;;
    *) echo 'Refusing credentials for an unexpected Git endpoint' >&2; exit 1 ;;
esac
