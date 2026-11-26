"""Compose and render Phase Shift v3.

Python 3.10+, FFmpeg with libx264 and AAC, and these Python packages:
    python -m pip install numpy scipy soundfile opencv-python
    python make_phase_shift_v3.py

The stereo master is the score's XY signal. The renderer reopens that master,
so changes to the score, geometry or mix are reflected in both deliverables.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import soundfile as sf
from scipy import signal


RATE = 48_000
FPS = 60
SIZE = 1080
BEAT = 60 / 96
BAR = 4 * BEAT
TAU = 2 * np.pi
TITLE = "Phase Shift v3"
OUT = Path(__file__).resolve().parent

# Beat within bar, MIDI pitch, gate in beats. The up-down leap in the opening
# phrase returns at bar 15. The final answer keeps the rhythm and falls to D.
THEME = (
    ((0, 74, .64), (.75, 77, .63), (1.5, 76, 1.22), (3, 69, .80)),
    ((0, 70, .63), (.75, 74, .63), (1.5, 72, 1.53), (3.5, 69, .40)),
    ((0, 77, .96), (1.25, 79, .46), (1.75, 77, .43), (2.25, 76, 1.49)),
    ((0, 74, .70), (1, 69, .94), (2.5, 74, 1.34)),
)
MELODY = (
    ((0, 62, 1.92), (2.5, 69, 1.12)),
    ((0, 65, 1.46), (2, 64, 1.04), (3.50, 69, .41)),
    *THEME,
    ((0, 74, .83), (1.25, 72, .64), (2, 76, 1.10), (3.45, 77, .45)),
    ((0, 77, .51), (.65, 74, .64), (1.5, 70, 1.38), (3.25, 72, .49)),
    ((0, 67, .82), (1, 74, .67), (1.75, 69, 1.82)),
    ((0, 73, 1.20), (1.5, 76, .64), (2.5, 69, 1.19)),
    ((0, 69, 1.85), (2.5, 72, 1.23)),
    ((0, 67, 1.46), (2.25, 64, 1.52)),
    ((0, 67, .76), (1, 70, .83), (2, 74, 1.72)),
    ((0, 73, .70), (1, 76, .74), (2, 79, .66), (3, 81, .79)),
    *THEME[:3],
    ((0, 74, .70), (1, 69, .64), (1.75, 73, .58), (2.5, 76, 1.20)),
    ((0, 77, .67), (.75, 81, .66), (1.5, 79, .87), (2.75, 76, 1.12)),
    ((0, 75, .62), (.75, 76, .61), (1.5, 73, 1.63), (3.5, 69, .39)),
    ((0, 74, .74), (1, 77, .73), (2, 76, 1.33)),
    ((0, 70, .78), (1.25, 74, .73), (2.25, 72, 1.37)),
    ((0, 69, 1.32), (1.75, 64, .68), (2.75, 61, .87)),
    ((0, 62, 3.54),),
)

# Bass, fifth or inversion, and three voicing tones per bar.
DM = (38, 45, 53, 57, 64)
BB = (34, 41, 53, 57, 62)
GM = (43, 50, 53, 57, 58)
ASUS = (33, 40, 52, 55, 62)
A7 = (33, 40, 52, 55, 61)
F = (41, 48, 52, 57, 60)
C = (36, 43, 52, 55, 62)
HARMONY = (
    DM, (38, 45, 53, 57, 62), DM, BB, GM, ASUS,
    (36, 45, 53, 57, 62), BB, GM, A7, F, C,
    GM, A7, DM, BB, GM, A7, BB, A7,
    (41, 45, 53, 57, 62), BB, A7, (38, 45, 50, 53, 57),
)

def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def hz(note):
    return 440 * 2 ** ((note - 69) / 12)


def lead_controls(t):
    pitch = np.full(len(t), float(MELODY[0][0][1]))
    gate = np.zeros(len(t))
    accent = np.zeros(len(t))
    last_pitch = pitch[0] - 12
    notes = [(bar, beat, note, length) for bar, phrase in enumerate(MELODY)
             for beat, note, length in phrase]
    for bar, beat, note, length in notes:
        start = round((4 * bar + beat) * BEAT * RATE)
        if start >= len(t):
            break
        note -= 12
        glide = round(.011 * RATE)
        end = min(len(t), start + round((length * BEAT + .20) * RATE))
        age = t[start:end] - t[start]
        attack = ease(age / (.032 if beat == 0 else .026))
        release = ease((length * BEAT + .20 - age) / .26)
        weight = 1.03 if beat == 0 else .95
        gate[start:end] = np.maximum(gate[start:end], weight * attack * release
                                    * (.77 + .23 * np.exp(-2.8 * age)))
        accent[start:end] += weight * attack * np.exp(-age / .14) * release
        pitch[start:] = note
        pitch[start:start + glide] = last_pitch + (note - last_pitch) * ease(np.arange(glide) / glide)
        pitch[start:end] += .10 * ease((age - .25) / .38) * np.sin(TAU * 4.7 * age)
        last_pitch = note
    phase = np.cumsum(hz(pitch)) * (TAU / RATE)
    return phase, gate, accent


# Both bodies are scanned during one note period. Cube edges form a closed
# walk, so changing faces never sends the beam through the middle of a face.
TABLE = 768
BODY = 384
U = np.arange(BODY) / BODY
A = TAU * U
VERTICES = np.array([(x, y, z) for z in (-1, 1)
                     for y in (-1, 1) for x in (-1, 1)], dtype=float)
WALK = [0, 1, 5, 1, 3, 7, 5, 4, 0, 2, 3, 7, 6, 2, 6, 4, 0]


def progress(t, start, end):
    return ease((t - start) / (end - start))


def blend(a, b, amount):
    return a + amount * (b - a)


def rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def cube(yaw, tilt, depth, size, roll=0):
    vertices = VERTICES.copy()
    vertices[:, 2] *= depth
    vertices[:, [0, 2]] = rotate(vertices[:, [0, 2]], yaw)
    vertices[:, [1, 2]] = rotate(vertices[:, [1, 2]], tilt)
    xy = rotate(vertices[:, :2], roll) * size / (1 - .085 * vertices[:, 2:])
    path = xy[WALK]
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return np.column_stack([np.interp(U * length[-1], length, axis) for axis in path.T])


def ring(radius, flatness, angle):
    return rotate(np.column_stack((np.cos(A), flatness * np.sin(A))), angle) * radius


def globe(t, opening, radius, center=(0, 0), spin=0):
    # The same continuous equator expands into five turns across the sphere.
    longitude = TAU * 5 * U + spin
    latitude = (U - .5) * np.pi * opening
    radial = np.cos(latitude)
    xyz = np.column_stack((radial * np.cos(longitude),
                           radial * np.sin(longitude),
                           np.sin(latitude)))
    tilt = .58 + .11 * np.sin(.31 * t)
    y = xyz[:, 1] * np.cos(tilt) - xyz[:, 2] * np.sin(tilt)
    path = np.column_stack((xyz[:, 0], y))
    return rotate(path, .14 + .065 * (t - 34)) * radius + np.asarray(center)


def box_pair(t):
    extrusion = progress(t, 3.0, 8.1)
    unfold = .04 + .96 * progress(t, .65, 3.75)
    yaw = .14 + .75 * extrusion + .34 * (t - 8) * extrusion
    tilt = .09 + extrusion * (.43 + .16 * np.sin(.43 * t))
    depart = progress(t, 13.6, 19.6)
    swing = progress(t, 19.6, 25.4)
    size = .44 - .14 * depart
    primary = cube(yaw, tilt, extrusion, size, .09 + .14 * np.sin(.31 * t))
    primary[:, 1] *= unfold
    primary += [-.32 * depart, .15 * swing]
    emerge = progress(t, 7.5, 12.55)
    twin = cube(yaw - emerge * (.62 + .55 * np.sin(.50 * t)), tilt + .16,
                extrusion, size * (.99 - .50 * emerge + .26 * depart),
                .09 + .14 * np.sin(.31 * t) - .27 * depart)
    twin[:, 1] *= unfold
    twin += [.48 * depart - .28 * swing, .13 * depart - .47 * swing]
    return primary, blend(primary, twin, emerge)


def shell_pair(t):
    opening = progress(t, 34, 40)
    split = progress(t, 44, 49.5)
    orbit = progress(t, 49.5, 55.8)
    close = progress(t, 55.8, 58.0)
    extinguish = 1 - progress(t, 58, 59.95)
    center = np.array([-.18 * split * (1 - close), .08 * split * (1 - close)])
    primary = globe(t, opening * (1 - .97 * close), .58 * extinguish, center=center)
    satellite = globe(t, opening * .9, .25 * extinguish, center=(.03, -.02), spin=.54)
    departing = globe(t + 5, .86, (.25 + .07 * np.sin(np.pi * orbit)) * extinguish,
                      center=(.48 - .55 * orbit, -.16 + .46 * np.sin(np.pi * orbit)),
                      spin=.54 + .2 * orbit)
    secondary = blend(satellite, departing, split)
    secondary = blend(secondary, globe(t, opening, .27 * extinguish,
                                       center=center, spin=.54), close)
    return primary, secondary


def frame_waveform(t):
    if t < 29:
        first, second = box_pair(t)
    elif t < 34:
        boxes = box_pair(t)
        rings = (globe(t, 0, .58),
                 globe(t, 0, .29, center=(.04, -.02), spin=.54))
        amount = progress(t, 29, 34)
        first = blend(boxes[0], rings[0], amount)
        second = blend(boxes[1], rings[1], amount)
    else:
        first, second = shell_pair(t)
    return np.vstack((first, second))


def dust_amount(t):
    return progress(t, 24.7, 28.8) * (1 - progress(t, 32.0, 35.15))


def lead_points(t):
    phase, gate, accent = lead_controls(t)
    phase_index = phase * TABLE / TAU
    index = np.floor(phase_index).astype(int) % TABLE
    fraction = (phase_index % 1)[:, None]
    result = np.empty((len(t), 2), np.float32)
    block = RATE // FPS
    current = frame_waveform(0)
    for start in range(0, len(t), block):
        end = min(start + block, len(t))
        following = frame_waveform(end / RATE)
        key = index[start:end]
        next_key = (key + 1) % TABLE
        first = blend(current[key], current[next_key], fraction[start:end])
        last = blend(following[key], following[next_key], fraction[start:end])
        result[start:end] = blend(first, last, (np.arange(end - start) / block)[:, None])
        current = following

    amount = dust_amount(t)
    active = amount > 0
    p = phase[active]
    # Slowly beating partials fill the exposure with grains, with no broadband
    # jumps between distant points. The stereo fundamentals carry the note.
    grains = np.column_stack((
        .30 * np.cos(p) + .21 * np.sin(1.506 * p) + .15 * np.cos(3.013 * p)
        + .11 * np.sin(7.73 * p) + .08 * np.cos(11.41 * p),
        .31 * np.sin(1.003 * p) + .22 * np.cos(2.498 * p) + .14 * np.sin(4.013 * p)
        + .10 * np.cos(8.17 * p) + .08 * np.sin(12.37 * p)))
    whirl = .58 * np.sin(.93 * t[active])
    c, s = np.cos(whirl), np.sin(whirl)
    x, y = grains.T
    grains = np.column_stack((x * c - y * s, x * s + y * c))
    result[active] = blend(result[active], grains, amount[active, None])
    return result * (.86 + .13 * gate + .01 * accent)[:, None]


def tone(mix, beat, length, note, gain, pan, attack, release, overtone):
    start = round(beat * BEAT * RATE)
    if start >= len(mix):
        return
    count = min(len(mix) - start, round(length * BEAT * RATE))
    t = np.arange(count) / RATE
    phase = TAU * hz(note) * t + .035*np.sin(TAU*.17*t)
    envelope = ease(t / attack) * ease((length * BEAT - t) / release)
    wave = (np.sin(phase) + overtone*np.sin(2*phase+.3)
            + .12*np.sin(3*phase+.7) + .055*np.sin(4*phase+.2)) * (gain*envelope)
    mix[start:start + count, 0] += wave * (1 - pan)
    mix[start:start + count, 1] += wave * (1 + pan)


def hit(mix, beat, strength, kind, rng):
    start = round(beat * BEAT * RATE)
    if start >= len(mix):
        return
    count = min(len(mix) - start, round(.31 * RATE))
    t = np.arange(count) / RATE
    if kind == "kick":
        phase = TAU * (46 * t + 2.2 * (1 - np.exp(-41 * t)))
        wave = np.sin(phase) * np.exp(-15 * t)
        pan = .06
    else:
        noise = rng.uniform(-1, 1, count)
        high = np.diff(noise, prepend=0)
        if kind == "clap":
            bounce = np.exp(-((t - .016) / .005) ** 2)
            wave = (.57 * high + .44 * np.sin(TAU * 171 * t)) * np.exp(-30 * t) + .27 * high * bounce
            pan = -.15
        else:
            wave = high * np.exp(-56 * t)
            pan = .24
    wave *= strength * ease(t / .002) * ease((.31 - t) / .075)
    mix[start:start + count, 0] += wave * (1 - pan)
    mix[start:start + count, 1] += wave * (1 + pan)


def accompaniment(mix):
    rng = np.random.default_rng(46)
    for bar, (root, fifth, low, middle, high) in enumerate(HARMONY):
        at = 4 * bar
        if 2 <= bar < 20:
            for when, pitch, length, gain in ((0, root, 1.38, .072), (1.75, fifth, .48, .037),
                                              (2.5, root, 1.16, .057)):
                tone(mix, at + when, length, pitch, gain, .07, .018, .15, .23)
            percussion = .62 if 10 <= bar < 14 else .74 if 18 <= bar < 20 else 1
            hit(mix, at, .094 * percussion, "kick", rng)
            hit(mix, at + 2.5, .070 * percussion, "kick", rng)
            hit(mix, at + 1, .027 * percussion, "clap", rng)
            hit(mix, at + 3, .035 * percussion, "clap", rng)
            for step, volume in ((.5, .014), (1.5, .024), (2.25, .016), (3.5, .023)):
                hit(mix, at + step, volume * percussion, "hat", rng)
        else:
            tone(mix, at, 3.80 if bar < 23 else 3.55, root, .037, .08, .14, .37, .19)

        for when, pitch, pan in ((.12, low, -.29), (1.00, middle, .16), (2.20, high, .38)):
            if bar == 23 and when > 1:
                continue
            gain = .022 if 10 <= bar < 14 else .014
            tone(mix, at + when, 2.0, pitch, gain, pan, .08, .66, .16)
        if 14 <= bar < 20:
            # A short answering note leaves the main melody exposed on beat one.
            tone(mix, at + 3.25, .66, high + 12, .027, -.45, .03, .26, .34)
        if bar in (1, 9, 13, 17, 19):
            hit(mix, at + 3.5, .032, "hat", rng)


def air_bursts(mix):
    rng = np.random.default_rng(202)
    sos = signal.butter(3, [7000, 16500], btype="bandpass", fs=RATE, output="sos")
    count = round(.038*RATE)
    envelope = np.exp(-np.arange(count)/(RATE*.013))*ease(np.arange(count)/(RATE*.0015))
    for bar in range(24):
        if not (0 <= bar < 4 or 10 <= bar < 14 or 22 <= bar < 24):
            continue
        gain = .027 if 10 <= bar < 14 else .016
        for beat in (.5, 1.5, 2.75, 3.5):
            start = round((bar*4+beat)*BEAT*RATE)
            end = min(len(mix), start+count)
            if start >= len(mix):
                break
            noise = rng.normal(0, 1, (end-start, 2))
            burst = signal.sosfilt(sos, noise, axis=0)*envelope[:end-start, None]*gain
            mix[start:end] += burst


def master(path, duration):
    t = np.arange(round(duration * RATE)) / RATE
    xy = lead_points(t)
    accompaniment(xy)
    air_bursts(xy)
    xy *= (ease(t / .041) * ease((len(t) / RATE - t) / .17))[:, None]
    peak = max(np.abs(signal.resample_poly(xy[max(0,i-32):i+RATE+32], 4, 1, axis=0)).max()
               for i in range(0,len(xy),RATE))
    xy *= 10 ** (-1.65 / 20) / peak
    sf.write(path, xy, RATE, subtype="PCM_24")


class Scope:
    """Render one sample-cycle as phosphor dots without frame-to-frame trails."""

    def __init__(self):
        self.core = np.zeros((SIZE, SIZE, 3), np.uint8)

    def draw(self, xy, time):
        self.core.fill(0)
        points = np.rint(xy*np.array([1, -1])*(SIZE*.66)+SIZE/2).astype(np.int32)
        cloud = 25 <= time < 33
        for i, (x, y) in enumerate(points):
            if cloud:
                color = (146, 207, 241) if i % 2 else (189, 226, 247)
            else:
                color = (112, 211, 185) if i % 2 else (211, 246, 226)
            radius = 3 if i % 12 == 0 else 2
            cv2.circle(self.core, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        halo = cv2.GaussianBlur(self.core, (0, 0), 1.45)
        return cv2.addWeighted(halo, .36, self.core, 1.0, 0)


def render(wav, movie):
    xy, rate = sf.read(wav, dtype="float32", always_2d=True)
    if rate != RATE or xy.shape[1] != 2 or len(xy) % (rate // FPS):
        raise ValueError("Expected frame-aligned 48 kHz stereo PCM")

    stride = rate // FPS
    scope = Scope()
    t = np.arange(len(xy))/rate
    phase = lead_controls(t)[0]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SIZE}x{SIZE}",
               "-framerate", str(FPS), "-i", "-", "-i", str(wav),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast",
               "-threads", "3", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "384k",
               "-movflags", "+faststart", "-metadata", f"title={TITLE}", str(movie)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for i in range(0, len(xy), stride):
            end_frame = min(i+stride, len(xy))
            cycle = np.ceil(phase[i]/TAU)*TAU
            start = np.searchsorted(phase, cycle, side="left")
            if start >= end_frame:
                start = i
                cycle = phase[start]
            end = np.searchsorted(phase, cycle+TAU, side="left")
            end = min(max(end, start+2), end_frame)
            indexes = np.linspace(start, end-1, 240)
            cycle_xy = np.column_stack([
                np.interp(indexes, np.arange(start, end), xy[start:end, channel])
                for channel in range(2)
            ])
            frame = scope.draw(cycle_xy, i/rate)
            process.stdin.write(frame.tobytes())
            if i % (stride * FPS * 10) == 0:
                print(f"Rendered {i // (stride * FPS):02d} seconds", flush=True)
        process.stdin.close()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audio-only", action="store_true", help="Compose the WAV")
    modes.add_argument("--video-only", action="store_true", help="Render from the existing WAV")
    parser.add_argument("--excerpt", action="store_true", help="Generate the opening 12.5 seconds")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    title = TITLE + (" excerpt" if args.excerpt else "")
    wav = args.output_dir / f"{title}.wav"
    movie = wav.with_suffix(".mp4")
    with TemporaryDirectory(dir=args.output_dir, prefix=".phase-shift-") as scratch:
        stage = Path(scratch)
        staged_wav, staged_movie = stage / wav.name, stage / movie.name
        if args.video_only:
            render(wav, staged_movie)
            staged_movie.replace(movie)
            return

        master(staged_wav, 12.5 if args.excerpt else len(MELODY) * BAR)
        if args.audio_only:
            staged_wav.replace(wav)
            return

        # Finish both encodes before publishing. Restore the previous WAV if a
        # player holds the destination movie open during the last rename.
        render(staged_wav, staged_movie)
        previous = stage / "previous.wav"
        if wav.exists():
            shutil.copyfile(wav, previous)
        staged_wav.replace(wav)
        try:
            staged_movie.replace(movie)
        except OSError:
            if previous.exists():
                previous.replace(wav)
            else:
                wav.unlink()
            raise


if __name__ == "__main__":
    main()
