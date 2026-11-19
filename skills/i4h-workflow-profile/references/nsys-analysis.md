# Nsight Systems Analysis for Workflow Runs

These commands target Nsight Systems 2024.5 or newer. Run them from the run directory, against the report that the profiled launch wrote.

## Export the reports

```bash
cd "$RUN_DIR"
mkdir -p nsys
nsys stats --force-export=true --format csv --output nsys/stats \
  --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  profile.nsys-rep
nsys analyze --format csv --output nsys/rule \
  --rule cuda_memcpy_sync --rule cuda_memcpy_async --rule cuda_api_sync --rule gpu_gaps --rule gpu_time_util \
  profile.sqlite
python3 - profile.sqlite "$I4H_WORKFLOWS/skills/i4h-workflow-profile/references/gpu_by_process.sql" <<'PY' > nsys/gpu_by_process.csv
import csv, sqlite3, sys
rows = sqlite3.connect(sys.argv[1]).execute(open(sys.argv[2]).read())
writer = csv.writer(sys.stdout)
writer.writerow([column[0] for column in rows.description])
writer.writerows(rows)
PY
```

`nsys stats` writes one `nsys/stats_<report>.csv` per report and leaves `profile.sqlite` beside the report. `nsys analyze` writes one `nsys/rule_<rule>.csv` per rule, and an empty file means the rule found nothing. [`gpu_by_process.sql`](gpu_by_process.sql) totals kernel time per process and labels the process that emitted the i4h ranges as `simulator`. Name the comparison capture's directory differently, for example `nsys-after/`.

## Read the reports

| Report | Columns to use | Question it answers |
|---|---|---|
| `nvtx_pushpop_sum` | `Range`, `Instances`, `Avg (ns)`, `Med (ns)`, `StdDev (ns)` | Which phase costs the most wall time per step, and how stable is it? |
| `nvtx_gpu_proj_sum` | `Range`, `Total Range Time (ns)`, `Total Proj Time (ns)`, `Avg GPU Ops` | How much GPU work did each phase launch? |
| `cuda_gpu_kern_sum` | `Name`, `Instances`, `Total Time (ns)` | Which kernels the GPU time goes to, across all captured processes. |
| `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum` | `Operation`, `Count`, `Total Time (ns)`, `Total (MB)` | How much time and volume host and device copies take. |
| `cuda_api_sum` | `Name`, `Num Calls`, `Total Time (ns)` | Which CUDA calls block the CPU thread. |
| `rule_cuda_api_sync`, `rule_cuda_memcpy_sync` | `Duration (ns)`, `API Name` | Where the CPU waited on the GPU. |
| `rule_gpu_gaps`, `rule_gpu_time_util` | gap duration and utilization per interval | Whether the GPU sat idle between steps. |
| `gpu_by_process` | `pid`, `role`, `kernels`, `gpu_ms` | How GPU time splits between the simulator and a backend. |

Range names carry a leading `:` in these reports, for example `:env.step`. Every range should have about one instance per captured step. Counts can differ by one, because the capture opens partway through a step and the run can end partway through one. `record` is absent from runs without `--record`.

## Classify a phase

- **GPU-bound:** projected GPU time is close to range time. Look at the heaviest kernels in that phase.
- **CPU-bound:** range time is far above projected GPU time, and `cuda_api_sum` shows little blocking. The cost is Python or host work; add `osrt` to `--trace` if the cause is not visible in the code.
- **Synchronization-bound:** range time is far above projected GPU time, and the time sits in `cudaDeviceSynchronize`, `cudaStreamSynchronize`, or synchronous `cudaMemcpy` calls. A tensor `.cpu()`, `.item()`, `.numpy()`, or `.tolist()` on a device tensor causes these.
- **Transfer-bound:** device-to-host or host-to-device copies dominate `cuda_gpu_mem_time_sum` within the phase.
- **Waiting on a backend:** `tick` is long with almost no GPU projection while a remote policy is active, and `gpu_by_process` shows the GPU time in an `other` process. The time is the backend's inference and round trip; check its `backend-*.log` and the request cadence.

## Common findings

| Finding | Usual owned fix |
|---|---|
| A Task pulls device tensors to the host every step | Keep the computation on the device, or read the host copy once per step and share it. |
| `record` performs one copy per field per step | Batch fields into one transfer or stage them in pinned memory; keep the HDF5 contract unchanged. |
| Step time not covered by the three ranges is large in a visible run | Measure headless before optimizing; sensor windows and the viewport render between ranges. |
| A sensor render dominates `env.step` | Add a `phase` range around the sensor update to confirm, check the selected backend and device in the run log, and avoid re-uploading unchanged data each step. |
| A remote policy is requested every step but its action chunk spans several | Consume the whole chunk before the next request, as the Task manifest declares. |
| GPU gaps between short kernels in `env.step` | Report them; physics and renderer scheduling belong to Isaac Sim, so route any setting change through the scene-edit skill. |

## Compare two captures

Compare the same range's `Avg (ns)` and `Med (ns)` from both `stats_nvtx_pushpop_sum.csv` files. Treat a difference smaller than either capture's `StdDev (ns)` as noise. Report both means, the difference in milliseconds per step, and the percentage change for every phase, not only the target.
