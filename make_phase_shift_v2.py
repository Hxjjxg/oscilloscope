"""Compose and render Phase Shift v2 next to this script.

Use Python 3.10 or newer and put FFmpeg with libx264 on PATH.
    python -m pip install numpy soundfile opencv-python
    python make_phase_shift_v2.py
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf


RATE = 48_000
FPS = 60
SIZE = 900
BEAT = 0.5
BARS = 22
TAU = 2 * np.pi
OUT = Path(__file__).resolve().parent

# Beat offset, MIDI note, gate in beats. Each entry begins on the upbeat.
HOOK = (
    (0.5, 76, 0.42), (1, 78, 0.68), (1.75, 81, 0.66),
    (2.5, 80, 0.90), (3.5, 76, 0.42), (4, 73, 0.89),
    (5, 71, 0.42), (5.5, 73, 0.89), (6.5, 69, 1.40),
)
HOOK_BARS = (1, 5, 13, 17)

PHRASES = {
    0: ((0, 57, 1.38), (1.75, 64, 0.69), (2.75, 71, 1.0)),
    3: ((0, 73, 1.33), (1.5, 69, 0.89), (2.75, 66, 1.14),
        (4, 68, 1.30), (5.5, 71, 0.42), (6, 76, 1.22)),
    7: ((0, 66, 2.68), (3, 61, 0.87), (4, 64, 1.65), (6, 59, 1.72)),
    9: ((0, 62, 1.8), (2.25, 66, 0.58), (3, 69, 0.9),
        (4.5, 68, 1.32), (6, 64, 1.49)),
    11: ((0, 61, 1.67), (2, 66, 1.32), (3.5, 69, 0.43),
         (4, 71, 1.38), (5.5, 73, 0.87), (6.5, 76, 1.29)),
    15: ((0, 78, 0.85), (1.25, 76, 0.66), (2, 73, 1.60),
         (4, 74, 0.88), (5, 71, 0.65), (5.75, 68, 1.61)),
    19: ((0, 73, 1.58), (2, 71, 0.91), (3, 64, 0.93),
         (4, 66, 1.41), (5.5, 64, 0.89), (6.5, 61, 1.43), (8, 57, 3.95)),
}

# Root and two color tones per bar. The F-sharp minor bridge shifts the register down.
A, E, FS, D, B = (33, 57, 61), (40, 56, 59), (30, 57, 61), (38, 54, 57), (35, 54, 62)
CHORDS = (A, A, D, FS, E, A, D, FS, FS, D, E, FS, E, A, D, FS, B, A, D, A, E, A)


def ease(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def hz(note):
    return 440 * 2 ** ((note - 69) / 12)


def oscillator(pitch):
    return np.cumsum(hz(pitch)) * (TAU / RATE)


def lead_score():
    phrases = PHRASES | dict.fromkeys(HOOK_BARS, HOOK)
    return sorted((bar * 4 + beat, note, gate)
                  for bar, phrase in phrases.items() for beat, note, gate in phrase)


def lead_controls(time):
    pitch = np.full(len(time), 57.0)
    envelope = np.zeros(len(time))
    accent = np.zeros(len(time))
    for beat, note, gate in lead_score():
        start = round(beat * BEAT * RATE)
        end = min(len(time), round((beat + gate) * BEAT * RATE))
        age = time[start:end] - time[start]
        pitch[start:] = note
        pitch[start:end] += 0.12 * np.sin(TAU * 5.1 * age) * ease((age - 0.22) / 0.24)
        weight = 1.07 if note in (81, 69) and beat / 4 >= 1 else 1.0
        attack = ease(age / 0.025)
        release = ease((gate * BEAT - age) / 0.042)
        envelope[start:end] = weight * attack * release * (0.79 + 0.21 * np.exp(-8 * age))
        accent[start:end] = weight * attack * release * np.exp(-5.4 * age)
    return oscillator(pitch), envelope, accent


def geometry(time, phase, accent):
    # All shapes retain a fundamental at the lead pitch. The slow morphs
    # change the timbre by moving energy between its harmonics.
    cos, sin = np.cos(phase), np.sin(phase)
    radius = 1 + (0.07 + 0.16 * accent) * np.cos(4 * phase + 0.7 * np.sin(time))
    petals = np.column_stack((0.86 * radius * cos, 0.72 * radius * sin))

    fold_x = 0.69 * cos + 0.25 * np.cos(3 * phase)
    fold_y = 0.49 * sin + 0.30 * np.sin(3 * phase)
    ribbon = np.column_stack((fold_x, fold_y))

    angle = np.remainder(phase, TAU)
    corners = np.arange(5) * (TAU / 4)
    diamond = np.column_stack((
        np.interp(angle, corners, [1, 0, -1, 0, 1]),
        np.interp(angle, corners, [0, 0.83, 0, -0.83, 0]),
    ))
    diamond[:, 1] += (0.15 + 0.17 * accent) * np.sin(2 * phase)

    azimuth = 0.38 * np.sin(TAU * (time - 26) / 7)
    depth = 0.42 * np.sin(2 * phase)
    orbit = np.column_stack((
        0.75 * cos * np.cos(azimuth) + depth * np.sin(azimuth),
        0.61 * sin + 0.31 * np.sin(3 * phase + 0.55 * np.sin(time * 0.8)),
    ))
    orbit /= (1 + 0.32 * depth[:, None])

    path = petals.copy()
    for start, shape in ((6, diamond), (10, petals), (14, ribbon),
                         (26.13, orbit), (30, diamond), (34.13, orbit), (38, petals)):
        path += ease((time - start) / 0.30)[:, None] * (shape - path)
    # The shape at the end of every hook opens toward a stable, upright oval.
    oval = np.column_stack((0.97 * cos, 0.74 * sin))
    for bar in HOOK_BARS:
        arrival = (bar * 4 + HOOK[-1][0]) * BEAT
        opening = ease((time - arrival) / 0.16) * ease((arrival + 0.83 - time) / 0.17)
        path += opening[:, None] * (oval - path)

    rotation = 0.26 * np.sin(TAU * time / 8) + 0.10 * accent
    x, y = path.T
    return np.column_stack((x * np.cos(rotation) - y * np.sin(rotation),
                            x * np.sin(rotation) + y * np.cos(rotation)))


def accompaniment(time):
    bar = (time / (4 * BEAT)).astype(int)
    pos = time / BEAT - bar * 4
    root, third, fifth = np.asarray(CHORDS)[bar].T
    offbeat = pos >= 2.75
    bass_age = (pos - np.where(offbeat, 2.75, np.where(pos >= 2, 2, 0))) * BEAT
    bass = np.sin(oscillator(root + 7 * offbeat))
    bass *= ease(bass_age / 0.035) * np.exp(-5.0 * bass_age)
    # Taper before changing pitch or reattacking the same note.
    bass *= ease((np.where(offbeat, 4, np.where(pos >= 2, 2.75, 2)) - pos) * BEAT / 0.038)

    bridge = ease((time - 14) / 0.55) * ease((26 - time) / 0.55)
    chord_gate = ease(pos * BEAT / 0.15) * ease((4 - pos) * BEAT / 0.19)
    chord = chord_gate * (0.012 + 0.015 * bridge)
    chords = np.column_stack((
        0.85 * np.sin(oscillator(third)) + 0.50 * np.sin(oscillator(fifth)),
        0.70 * np.cos(oscillator(third)) - 0.70 * np.cos(oscillator(fifth)),
    ))
    late = ease((time - 30) / 2) * ease((39 - time) / 1)
    tick_age = np.remainder(time, BEAT / 2)
    tick = np.sin(TAU * (1280 * tick_age + 11 * (1 - np.exp(-60 * tick_age))))
    tick *= ease(tick_age / 0.006) * np.exp(-100 * tick_age) * (0.010 + 0.015 * late)
    bass_level = 0.104 - 0.036 * late
    return ((bass_level * bass)[:, None] * [1, 0.62]
            + chord[:, None] * chords + tick[:, None] * [0.58, -1])


def compose():
    time = np.arange(round(BARS * 4 * BEAT * RATE)) / RATE
    phase, envelope, accent = lead_controls(time)
    lead = geometry(time, phase, accent) * (0.86 * envelope[:, None])
    mix = lead + accompaniment(time)
    mix *= (ease(time / 0.12) * ease((time[-1] - time) / 1.28))[:, None]
    mix *= 0.84 / np.max(np.abs(mix))
    return mix


def graticule():
    axis = np.linspace(-1, 1, SIZE, dtype=np.float32)
    vignette = np.clip(1 - (axis[:, None] ** 2 + axis ** 2) * 0.60, 0, 1)
    grid = np.uint8(np.array([11, 9, 7]) + vignette[:, :, None] * [13, 9, 6])
    inset = 90
    for pos in range(inset, SIZE - inset + 1, 90):
        color = (45, 37, 30) if pos == SIZE // 2 else (31, 26, 22)
        cv2.line(grid, (inset, pos), (SIZE - inset, pos), color, 1)
        cv2.line(grid, (pos, inset), (pos, SIZE - inset), color, 1)
        cv2.circle(grid, (SIZE // 2, pos), 2, (55, 46, 38), -1, cv2.LINE_AA)
        cv2.circle(grid, (pos, SIZE // 2), 2, (55, 46, 38), -1, cv2.LINE_AA)
    return grid


def render(wav, video):
    cv2.setNumThreads(1)
    samples, rate = sf.read(wav, dtype="float32")
    frame_length = RATE // FPS
    if rate != RATE or samples.shape[1] != 2 or len(samples) % frame_length:
        raise ValueError("Render input needs to be frame-aligned, 48 kHz stereo.")
    grid = graticule()
    phosphor = np.zeros((SIZE, SIZE), dtype=np.uint8)
    trace = np.zeros_like(phosphor)
    decay = np.uint8(np.arange(256) * 0.19)
    commands = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{SIZE}x{SIZE}",
        "-framerate", str(FPS), "-i", "pipe:0", "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast",
        "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart", str(video),
    ]
    with subprocess.Popen(commands, stdin=subprocess.PIPE) as process:
        try:
            for start in range(0, len(samples), frame_length):
                xy = samples[max(0, start - 1):start + frame_length]
                points = np.rint((SIZE / 2 + xy * [417, -417]) * 16).astype(np.int32)
                trace.fill(0)
                level = int(np.interp(np.max(np.abs(xy)), [0.015, 0.085, 0.60], [0, 68, 222]))
                cv2.polylines(trace, [points], False, level, 1, cv2.LINE_AA, shift=4)
                cv2.LUT(phosphor, decay, dst=phosphor)
                cv2.max(phosphor, trace, dst=phosphor)
                small = cv2.resize(phosphor, (225, 225), interpolation=cv2.INTER_AREA)
                bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.35), (SIZE, SIZE))
                beam = cv2.merge([
                    cv2.addWeighted(phosphor, 1.02, bloom, 2.0, 0),
                    cv2.addWeighted(phosphor, 0.93, bloom, 1.35, 0),
                    cv2.addWeighted(phosphor, 0.62, bloom, 0.46, 0),
                ])
                process.stdin.write(cv2.add(grid, beam).tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, commands)


def main():
    wav = OUT / "Phase Shift v2.wav"
    video = OUT / "Phase Shift v2.mp4"
    samples = compose()
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 1:
        raise ValueError("Master contains invalid or clipping samples.")
    sf.write(wav, samples, RATE, subtype="PCM_24")
    render(wav, video)


if __name__ == "__main__":
    main()
