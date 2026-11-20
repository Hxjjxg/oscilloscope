"""Generate the Phase Shift stereo master and a preview drawn from that master."""

from pathlib import Path
import subprocess

import cv2
import numpy as np
import soundfile as sf


SAMPLE_RATE = 48_000
FPS = 60
SIZE = 720
BEAT = 0.5
TAU = 2 * np.pi
OUTPUT = Path(__file__).resolve().parent

# D minor, B-flat, F, C. Each position lasts an eighth note.
MELODY = np.array([
    [62, 69, 65, 72, 74, 69, 65, 69],
    [58, 65, 62, 69, 70, 65, 62, 65],
    [60, 67, 65, 72, 77, 72, 69, 65],
    [60, 67, 64, 70, 72, 67, 64, 69],
    [62, 74, 69, 77, 76, 72, 69, 65],
    [58, 70, 65, 74, 72, 69, 65, 62],
    [65, 77, 72, 81, 79, 76, 72, 69],
    [60, 72, 67, 76, 74, 70, 69, 67],
    [62, 69, 74, 77, 81, 77, 74, 69],
    [58, 65, 70, 74, 77, 74, 70, 65],
    [65, 72, 77, 81, 84, 81, 77, 72],
    [60, 67, 72, 76, 74, 69, 62, 62],
])

# Every octahedron vertex has degree four, so this closed path visits each
# edge once without jumping across any face or retracing an edge.
VERTICES = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
])
OCTAHEDRON = VERTICES[[0, 2, 1, 3, 0, 4, 2, 5, 3, 4, 1, 5, 0]]


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def frequency(midi):
    return 440 * 2 ** ((midi - 69) / 12)


def walk(points, phase):
    position = (phase % TAU) * ((len(points) - 1) / TAU)
    index = position.astype(int)
    fraction = (position - index)[:, None]
    return points[index] + fraction * (points[index + 1] - points[index])


def draw_signal():
    eighth = BEAT / 2
    duration = 4 * BEAT * len(MELODY)
    time = np.arange(round(duration * SAMPLE_RATE)) / SAMPLE_RATE
    step = (time / eighth).astype(int)
    bar = step // 8
    step_time = time - step * eighth
    beat_time = time % BEAT
    midi = MELODY.ravel()
    previous = midi[np.maximum(step - 1, 0)]
    pitch = previous + (midi[step] - previous) * smoothstep(step_time / 0.018)
    pitch += 0.13 * np.sin(TAU * 5 * time) * smoothstep(step_time / 0.22)
    phase = TAU / SAMPLE_RATE * np.cumsum(frequency(pitch))

    # The curve used to draw the picture is also the oscillator's waveform.
    angle = phase + 0.33 * np.sin(TAU * time / 4)
    lobes = 0.30 * smoothstep((time - 1) / 3) * (0.75 + 0.25 * np.cos(TAU * time / 4))
    radius = 0.78 * (1 + lobes * np.cos(5 * angle))
    flower = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))

    corners = np.linspace(0, TAU, 7)
    hexagon = np.column_stack((np.cos(corners), np.sin(corners)))
    # Vary the rate along the outline to accent the offbeats without adding strokes.
    polygon = walk(hexagon, angle + 0.10 * np.sin(3 * angle + TAU * time))
    twist = TAU * time / 16
    cos_twist, sin_twist = np.cos(twist), np.sin(twist)
    px, py = polygon.T
    polygon = np.column_stack((
        px * cos_twist - 0.78 * py * sin_twist,
        px * sin_twist + 0.78 * py * cos_twist,
    ))

    vx, vy, vz = walk(OCTAHEDRON, phase).T
    azimuth = TAU * (time - 16) / 6
    elevation = 0.43 + 0.27 * np.sin(TAU * (time - 16) / 8)
    x = vx * np.cos(azimuth) + vz * np.sin(azimuth)
    z = vz * np.cos(azimuth) - vx * np.sin(azimuth)
    y = vy * np.cos(elevation) - z * np.sin(elevation)
    depth = vy * np.sin(elevation) + z * np.cos(elevation)
    wireframe = np.column_stack((x, y)) / (1.05 + 0.15 * depth[:, None])

    # Short crossfades preserve the continuity of the beam between scenes.
    first_mix = smoothstep((time - 7.82) / 0.36)[:, None]
    second_mix = smoothstep((time - 15.82) / 0.36)[:, None]
    path = (1 - first_mix) * flower + first_mix * polygon
    path = (1 - second_mix) * path + second_mix * wireframe

    accents = np.array([1, 0.83, 0.93, 0.80, 1, 0.83, 0.92, 0.84])
    accent = accents[(step - 1) % 8]
    accent += (accents[step % 8] - accent) * smoothstep(step_time / 0.027)
    tail = np.exp(-9 * eighth)
    pluck = tail + (np.exp(-9 * step_time) - tail) * smoothstep(step_time / 0.019)
    pulse = (0.74 + 0.25 * pluck) * accent
    sustain = smoothstep((time - (duration - 0.55)) / 0.12)
    pulse += (0.73 - pulse) * sustain
    fade = smoothstep(time / 0.12) * smoothstep((duration - time) / 0.52)

    # A quiet bass pulse also moves the beam. Keep it small so each sweep
    # still lands close to the previous one.
    roots = np.array([38, 34, 41, 36, 38, 34, 41, 36, 38, 34, 41, 38])
    bass_phase = TAU / SAMPLE_RATE * np.cumsum(frequency(roots[bar]))
    bass_envelope = smoothstep(beat_time / 0.018) * np.exp(-12 * beat_time)
    bass = 0.064 * np.sin(bass_phase) * bass_envelope

    return ((0.90 * pulse[:, None] * path + bass[:, None] * [1, 0.83])
            * fade[:, None])


def make_graticule():
    grid = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    grid[:] = (13, 12, 5)
    inset = 62
    divisions = np.linspace(inset, SIZE - inset, 9).astype(int)
    for coordinate in divisions:
        color = (26, 28, 15) if coordinate == SIZE // 2 else (19, 21, 12)
        cv2.line(grid, (inset, coordinate), (SIZE - inset, coordinate), color, 1)
        cv2.line(grid, (coordinate, inset), (coordinate, SIZE - inset), color, 1)
    return grid


def render_preview(wav, video):
    samples, rate = sf.read(wav, dtype="float32")
    assert rate == SAMPLE_RATE and samples.shape[1] == 2
    samples_per_frame = rate // FPS
    frames = samples.reshape(-1, samples_per_frame, 2)
    grid = make_graticule()
    afterglow = np.zeros((SIZE, SIZE), dtype=np.float32)
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{SIZE}x{SIZE}",
        "-framerate", str(FPS), "-i", "pipe:0", "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast",
        "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart", str(video),
    ], stdin=subprocess.PIPE)
    try:
        for index, samples_xy in enumerate(frames):
            if index:
                samples_xy = samples[index * samples_per_frame - 1:(index + 1) * samples_per_frame]
            points = np.rint((SIZE / 2 + samples_xy * [325, -325]) * 16).astype(np.int32)
            trace = np.zeros((SIZE, SIZE), dtype=np.uint8)
            cv2.polylines(trace, [points], False, 255, 1, cv2.LINE_AA, shift=4)
            if np.max(np.abs(samples_xy)) < 0.02:
                trace.fill(0)
            afterglow *= 0.37
            np.maximum(afterglow, trace, out=afterglow)
            small = cv2.resize(afterglow, (SIZE // 3, SIZE // 3), interpolation=cv2.INTER_AREA)
            halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.0), (SIZE, SIZE))
            result = cv2.add(grid, cv2.merge([
                cv2.convertScaleAbs(afterglow * 0.53 + halo * 0.90),
                cv2.convertScaleAbs(afterglow * 0.92 + halo * 1.95),
                cv2.convertScaleAbs(afterglow * 0.33 + halo * 0.42),
            ]))
            process.stdin.write(result.tobytes())
    finally:
        process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


def main():
    wav = OUTPUT / "Phase Shift.wav"
    video = OUTPUT / "Phase Shift.mp4"
    samples = draw_signal()
    assert np.isfinite(samples).all() and np.max(np.abs(samples)) < 1
    sf.write(wav, samples, SAMPLE_RATE, subtype="PCM_24")
    render_preview(wav, video)


if __name__ == "__main__":
    main()
