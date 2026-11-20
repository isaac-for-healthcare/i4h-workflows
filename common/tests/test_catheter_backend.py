import json

import h5py
import pytest

from i4h_common.catheter_backend import CONTINUATION_SCHEMA, recording_provenance, resolve_backend


def recording(tmp_path, native):
    path = tmp_path / "demos.hdf5"
    with h5py.File(path, "w") as file:
        file.create_group("data").attrs["env_args"] = json.dumps(native)
    return path


def test_fresh_default_stays_legacy_until_native_promotion():
    assert resolve_backend() == "legacy"
    assert resolve_backend("numerical") == "numerical"


@pytest.mark.parametrize("backend", ["numerical", "legacy"])
def test_replay_selects_backend_and_rejects_explicit_mismatch(tmp_path, backend):
    path = recording(
        tmp_path,
        dict(catheter_backend=backend, numerical_sim_revision="abc", continuation_schema_version=CONTINUATION_SCHEMA),
    )
    assert resolve_backend(recording=path) == backend
    with pytest.raises(ValueError, match="differs"):
        resolve_backend("legacy" if backend == "numerical" else "numerical", path)


def test_old_recording_requires_explicit_legacy(tmp_path):
    path = recording(tmp_path, {})
    for requested in (None, "numerical"):
        with pytest.raises(ValueError, match="explicit"):
            resolve_backend(requested, path)
    assert resolve_backend("legacy", path) == "legacy"


def test_conflicting_sidecar_is_rejected(tmp_path):
    path = recording(tmp_path, {"catheter_backend": "numerical"})
    (tmp_path / "run.json").write_text(json.dumps({"catheter_backend": "legacy"}))
    with pytest.raises(ValueError, match="conflicting"):
        recording_provenance(path)
