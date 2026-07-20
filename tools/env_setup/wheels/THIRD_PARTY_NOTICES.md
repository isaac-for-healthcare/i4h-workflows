# Third-Party Software in this Directory

This directory contains pre-built binary wheels of third-party open source
software, vendored via Git LFS for users who cannot build them locally
(no public wheel exists for the target torch / CUDA / Python / SM matrix).

The build process is reproducible via [`../build_flash_attn.sh`](../build_flash_attn.sh).

---

## flash-attn

- **Project**: flash-attn
- **Upstream**: <https://github.com/Dao-AILab/flash-attention>
- **Source distribution**: <https://files.pythonhosted.org/packages/source/f/flash_attn/flash_attn-2.7.4.post1.tar.gz>
- **Version**: 2.7.4.post1
- **License**: BSD 3-Clause
- **Copyright**: Copyright (c) 2022, the respective contributors, as shown by
  the `AUTHORS` file in the upstream repository (primary author: Tri Dao,
  <trid@cs.stanford.edu>).
- **Full license text**: included inside each wheel as
  `flash_attn-2.7.4.post1.dist-info/LICENSE` and reproduced verbatim below.

### Build configuration

These wheels were built from the unmodified upstream source distribution with
two build-system patches applied by `../build_flash_attn.sh` (the patches do
not modify the FlashAttention algorithm or any runtime code; they only
extend the set of CUDA SM targets the build emits):

- Added `-gencode arch=compute_110,code=sm_110` for NVIDIA Jetson Thor.
- Added `-gencode arch=compute_121,code=sm_121` for NVIDIA DGX Spark.
- Default `FLASH_ATTN_CUDA_ARCHS` extended from upstream's `"80;90;100;120"`
  to `"80;90;100;110;120;121"`.

Build environment:

- torch 2.9.0 + CUDA 13.0 (cu130 wheel index)
- Python 3.11
- Linux x86_64 and aarch64 (separate wheel per arch)

After building, the platform tag was rewritten from `linux_<arch>` to
`manylinux_2_35_<arch>` so that uv / pip will accept the wheels as
PEP 600 compatible binaries on the target distros.

### License text (BSD 3-Clause)

```text
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
