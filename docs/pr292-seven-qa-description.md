# Local replacement description for PR 292

Integrate Numerical Sim using clean physics commit `267f48a4356cd2bf4a6204501b9daa7a2ee5c15b`,
including seven public QA fixes: static-vessel containment caps and insertion entry selection
(6940633), haptics/clipping test races (6940611), parity reference setup (7048521), recording
resume documentation (7048496), GL camera direction (7048533), aorta tip-camera placement
(7107434) and the GL UI dependency (7107443).

Keep the legacy solver pin, default navigation backend and Isaac Lab/Newton/Warp versions.
Containment caps affect only the query mesh. The internal manager lifecycle, state ownership,
masked reset and continuation contracts are preserved. Setup, editable package paths,
lock metadata, RL export revision and qualification source checks use the same physics commit.

Defer 6940614 entirely: preserve the 192-node centreline, reach, 576-value observations,
`guide_tip` and existing procedure/baseline data. Keep 7011011's second-press failure separate.

Qualification is blocked. Liver capture succeeds (1/1) and fresh-process replay completes,
but the full-episode encoded trajectory exceeds an existing feature bound. The extended
eight-environment aorta sweep has a transient positive signed distance at 12 seconds.
Runtime/package, rendering and Arena CPU failures are retained. Reset/restore and fresh-process
continuation checks pass at 1/8/16 environments, including live acquired grasps. Native static-aorta
capture/replay completes 180 steps with zero measured observation and final particle error. See
[the source-specific qualification report](pr292-seven-qa-qualification.md) for passing,
failing, skipped and unavailable checks, fresh snapshot identities and replay results.
Historical qualification artifacts remain under their original identities. No training campaign
was run and no acceptance threshold changed. GB300 verification requires suitable hardware.

This description is prepared locally. Fresh remote setup cannot fetch the unpublished physics
commit until it is published. No PR or tracker was updated.
