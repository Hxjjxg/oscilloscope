Create and deliver a complete, original second version of Phase Shift, an oscilloscope music piece with a distinctive, hummable lead melody and animation encoded in the same stereo audio. Aim for 35–50 seconds. Preserve the first version and its generator. Build, render, verify, and refine v2 in this run.

## Context

- First version: `make_phase_shift.py`, `Phase Shift.wav`, and `Phase Shift.mp4`. If the renders are missing, regenerate them from the script.
- Local lossless references: Atom Delta's `references/oscillofun-wave.flac` and BUS ERROR Collective's `references/Primer Final.flac`. Use both to study how rhythmic phrasing and transitions coincide with changes in the XY trace. Inspect selected passages by plotting their stereo samples and analyzing timing or pitch where useful. Compose all musical and visual material for v2 from scratch.

## Creative direction

- Introduce a short lead hook within the first eight seconds. Give it a deliberate rhythm and a recognizable pitch contour, with a clear point of arrival. Bring it back at least twice, including once after a contrasting section. Keep its defining rhythm and contour audible when the surrounding timbre, harmony, or geometry changes.
- Give the hook room to stand out. Use contrast in register, pacing, or dynamics to shape the other sections. Synchronize a few clear visual changes with the hook's entrances, accents, or cadence.
- Write and arrange the music around the lead hook before finalizing the visuals. Shape the XY paths and supporting sounds together so that the melody stays prominent and the main outlines still read at normal playback speed. Choose the implementation and tools that best serve the piece.

## Deliverables and verification

- Save `Phase Shift v2.wav` as 48 kHz, 24-bit stereo PCM, with left mapped to X and right to Y. Verify its format and duration, and confirm that it has headroom and no clipping.
- Save `Phase Shift v2.mp4` with synchronized stereo sound and an XY animation rendered directly from the final WAV samples. Inspect the rendered frames at the actual viewing size, including the hook's entrances, a scene transition, and the densest passage. Revise and render again if the main shapes blur together.
- Deliver a reproducible generator or editable source with the render command and any dependencies. Keep v1 files intact.
- Verify the hook against the rendered audio. Report its notes and rhythm, the timestamps of its first appearance and returns, and how you checked that the lead is still distinguishable after mixing. Use playback review if available. If you relied on signal analysis, state that and report the evidence you measured.
- Finish with the file paths, how to regenerate the piece, and the checks you actually completed. Make reasonable, reversible decisions and carry the work through to the finished files.
