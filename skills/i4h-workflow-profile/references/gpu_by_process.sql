-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- GPU kernel time per captured process; the process that emitted the i4h phase ranges is the simulator.
WITH kernels AS (
  SELECT (globalPid >> 24) & 16777215 AS pid, COUNT(*) AS kernels, SUM("end" - start) AS ns
  FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1
), simulator AS (
  SELECT DISTINCT (globalTid >> 24) & 16777215 AS pid FROM NVTX_EVENTS
  WHERE text IN ('tick', 'env.step', 'record')
)
SELECT kernels.pid,
       CASE WHEN kernels.pid IN (SELECT pid FROM simulator) THEN 'simulator' ELSE 'other' END AS role,
       kernels.kernels, ROUND(kernels.ns / 1e6, 2) AS gpu_ms
FROM kernels ORDER BY gpu_ms DESC;
