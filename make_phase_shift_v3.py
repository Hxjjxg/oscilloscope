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


def bass_phase(t):
    # Every bass period contains one unmasked XY traversal. Hold the root
    # through the keyboard's rests and change it legato at the next bar.
    freq = np.empty(len(t), dtype=np.float64)
    prev = hz(HARMONY[0][0])
    for bar, chord in enumerate(HARMONY):
        start = round(bar * BAR * RATE)
        if start >= len(t):
            break
        end = min(len(t), round((bar + 1) * BAR * RATE))
        target = hz(chord[0])
        freq[start:end] = blend(prev, target, ease((t[start:end] - bar * BAR) / .07))
        prev = target
    return np.cumsum(freq) / RATE

# Both bodies are scanned during one bass period. Cube edges form a closed
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
    # Material coordinates keep each dot on its edge even when foreshortened.
    edge = U * (len(path) - 1)
    start = edge.astype(int)
    return blend(path[start], path[start + 1], (edge - start)[:, None])


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
    yaw = (.14 + .64 * extrusion + 1.06 * progress(t, 8.4, 13.4)
           + 1.52 * progress(t, 15.1, 22.7) + .36 * progress(t, 23.4, 27.5))
    tilt = .09 + extrusion * (.43 + .16 * np.sin(.43 * t))
    depart = progress(t, 13.6, 19.6)
    swing = progress(t, 19.6, 25.4)
    size = .44 - .14 * depart
    primary = cube(yaw, tilt, extrusion, size, .09 + .14 * np.sin(.31 * t))
    primary[:, 1] *= unfold
    primary += [-.32 * depart, .15 * swing]
    emerge = progress(t, 7.5, 12.55)
    twin = cube(yaw - emerge * (.62 + .57 * progress(t, 16, 21)), tilt + .16,
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
    if t < 34:
        first, second = box_pair(t)
        # A deterministic winding opens off the faces, sweeps through the
        # volume and contracts onto the two equators. All grains retain IDs.
        amount = progress(t, 24.5, 29)
        if amount > 0:
            angle = A + .18 * (t - 25)
            field = np.column_stack((.55 * np.cos(3 * angle) + .17 * np.sin(8 * A),
                                     .44 * np.sin(4 * angle) + .18 * np.cos(7 * A)))
            first = blend(first, field, amount)
            second = blend(second, rotate(field, .69), amount)
            resolve = progress(t, 29, 34)
            ring1, ring2 = shell_pair(t)
            first = blend(first, ring1, resolve)
            second = blend(second, ring2, resolve)
    else:
        first, second = shell_pair(t)
    return np.vstack((first, second))


SCAN_START = .10
SCAN_LENGTH = .43


def scan_voice(t, phase):
    # Enter from the centre, trace both bodies, then return to the centre
    # before the ensemble opens up in the rest of the bass cycle.
    fraction = phase % 1
    key = np.clip((fraction - SCAN_START) / SCAN_LENGTH, 0, 1) * (TABLE - 1)
    index = key.astype(int)
    weight = (key - index)[:, None]
    next_index = np.minimum(index + 1, TABLE - 1)
    block = RATE // FPS
    current = frame_waveform(0)
    result = np.empty((len(t), 2), np.float32)
    for i in range(0, len(t), block):
        j = min(i + block, len(t))
        following = frame_waveform(j / RATE)
        a, b = index[i:j], next_index[i:j]
        this = blend(current[a], current[b], weight[i:j])
        after = blend(following[a], following[b], weight[i:j])
        result[i:j] = blend(this, after, (np.arange(j - i) / block)[:, None])
        current = following
    draw = ease((fraction - .025) / .07) * (1 - ease((fraction - .54) / .08))
    return result * draw[:, None]


def add_note(mix, beat, beats, note, gain, pan, voice):
    start = round(beat * BEAT * RATE)
    sustain = beats * BEAT
    release = {"keys": .29, "reed": .31, "bass": .12, "bell": .46}[voice]
    count = min(len(mix) - start, round((sustain + release) * RATE))
    if count <= 0:
        return
    age = np.arange(count) / RATE
    p = TAU * hz(note) * age
    tail = 1 - ease((age - sustain) / release)
    if voice == "keys":
        # Tine attack, then a warm, rapidly darkening fundamental.
        attack = ease(age / .008)
        body = np.sin(p + 1.15 * np.exp(-age / .105) * np.sin(2*p))
        wave = attack * tail * np.exp(-age / .57) * (body + .17*np.sin(2.002*p))
    elif voice == "bell":
        wave = (.72*np.sin(p) + .22*np.sin(2.72*p) * np.exp(-age/.18)
                + .13*np.sin(4.08*p) * np.exp(-age/.085))
        wave *= ease(age/.004) * np.exp(-age/.24) * tail
    elif voice == "reed":
        p += .14 * ease(age/.28) * np.sin(TAU*4.8*age)
        wave = (np.sin(p) + .26*np.sin(3*p) + .09*np.sin(5*p))
        wave *= ease(age/.14) * tail * (.91 + .09*np.sin(TAU*1.3*age))
    else:
        wave = np.sin(p + .68*np.exp(-age/.10)*np.sin(p)) + .13*np.sin(2*p)
        wave *= ease(age/.009) * np.exp(-age/.48) * tail
    mix[start:start+count] += (gain * wave)[:, None] * np.sqrt([1-pan, 1+pan])


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


def ensemble(length):
    keys = np.zeros((length, 2), dtype=np.float32)
    rhythm = np.zeros_like(keys)
    answers = np.zeros_like(keys)
    rng = np.random.default_rng(46)
    for bar, (root, fifth, low, mid, high) in enumerate(HARMONY):
        at = 4 * bar
        for beat, note, gate in MELODY[bar]:
            add_note(keys, at + beat, gate, note - (12 if bar < 2 else 0),
                     .40 if beat == 0 else .34, -.20, "keys")
        driving = 2 <= bar < 10 or 14 <= bar < 22
        groove = ((0, root, .85), (1.75, fifth, .33), (2.55, root+12, .49))
        if not driving:
            groove = ((0, root, 2.3),)
        for beat, note, gate in groove:
            add_note(rhythm, at + beat, gate, note, .42, 0, "bass")
        if driving:
            hit(rhythm, at, .56, "kick", rng)
            hit(rhythm, at+2.5, .33, "kick", rng)
            hit(rhythm, at+1.03, .21, "clap", rng)
            hit(rhythm, at+3.03, .27, "clap", rng)
            for beat, gain in ((.55,.07), (1.54,.10), (2.56,.055), (3.55,.12)):
                hit(rhythm, at+beat, gain, "hat", rng)
        if bar in (5, 9, 13, 17, 21):
            for beat, gain in ((3.25,.07), (3.56,.10), (3.78,.16)):
                hit(rhythm, at+beat, gain, "clap", rng)
        if bar in (3, 5, 7, 9, 15, 17, 19, 21):
            for beat, note in ((2.82, high+12), (3.54, mid+12)):
                add_note(answers, at+beat, .24, note, .28, .52, "bell")
        if 10 <= bar < 14 or 16 <= bar < 22:
            add_note(answers, at+.45, 2.64, mid, .22, .34, "reed")
            add_note(answers, at+1.38, 1.84, high, .13, -.54, "reed")
        if bar == 23:
            add_note(answers, at+.68, 1.5, low+12, .15, .32, "bell")
    # A single quiet reflection opposite the keyboard opens up the phrase.
    lag = round(.158 * RATE)
    if length > lag:
        keys[lag:, 1] += .17 * keys[:-lag, 0]
    return keys, rhythm, answers


def master(path, duration):
    t = np.arange(round(duration * RATE)) / RATE
    phase = bass_phase(t)
    xy = .56 * scan_voice(t, phase)
    fraction = phase % 1
    # Reopen musical voices after the last edge. The raw stereo trace has
    # a clean traversal on every bass cycle, even on a snare hit.
    bed = 1 - ease((fraction - .025) / .065) * (1 - ease((fraction - .54) / .095))
    for stem in ensemble(len(t)):
        xy += stem * bed[:, None]


    xy *= (ease(t / .041) * ease((len(t) / RATE - t) / .17))[:, None]
    peak = max(np.abs(signal.resample_poly(xy[max(0,i-32):i+RATE+32], 4, 1, axis=0)).max()
               for i in range(0,len(xy),RATE))
    xy *= 10 ** (-1.65 / 20) / peak
    sf.write(path, xy, RATE, subtype="PCM_24")


# Fractional-phase sampling gives every dot a stable material coordinate.
# Averaging five neighbouring bass traversals reduces interpolation error at
# corners. It operates on the quantized stereo master, in signal coordinates.
DOTS = 256
DOT_PHASE = SCAN_START + SCAN_LENGTH * (np.arange(DOTS) + .5) / DOTS


def sampled_dots(xy, phase, middle):
    position = phase[middle] - SCAN_START - SCAN_LENGTH / 2
    cycle = int(np.floor(position))
    starts = np.clip(cycle + np.arange(-2, 4), 0, int(phase[-1] - .55))
    targets = starts[:, None] + DOT_PHASE
    right = np.clip(np.searchsorted(phase, targets), 1, len(xy) - 1)
    left = right - 1
    w = (targets - phase[left]) / (phase[right] - phase[left])
    samples = blend(xy[left], xy[right], w[:, :, None])
    kernel = np.array([1, 4, 6, 4, 1]) / 16
    left_scan = np.tensordot(kernel, samples[:-1], axes=1)
    right_scan = np.tensordot(kernel, samples[1:], axes=1)
    return blend(left_scan, right_scan, position - cycle)


class Scope:
    def __init__(self):
        self.core = np.zeros((SIZE, SIZE, 3), np.uint8)

    def draw(self, xy):
        self.core.fill(0)
        # Subpixel placement and equal luminance avoid sparkling edges.
        pixels = (xy * [1, -1] * (SIZE * 1.10) + SIZE / 2) * 16
        for x, y in np.rint(pixels).astype(int):
            cv2.circle(self.core, (x, y), 29, (212, 239, 223), -1, cv2.LINE_AA, 4)
        halo = cv2.GaussianBlur(self.core, (0, 0), 1.65)
        return cv2.addWeighted(self.core, .94, halo, .38, 0)


def render(wav, movie):
    xy, rate = sf.read(wav, dtype="float32", always_2d=True)
    if rate != RATE or xy.shape[1] != 2 or len(xy) % (rate // FPS):
        raise ValueError("Expected frame-aligned 48 kHz stereo PCM")

    stride = rate // FPS
    scope = Scope()
    t = np.arange(len(xy))/rate
    phase = bass_phase(t)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SIZE}x{SIZE}",
               "-framerate", str(FPS), "-i", "-", "-i", str(wav),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast",
               "-threads", "3", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "384k",
               "-movflags", "+faststart", "-metadata", f"title={TITLE}", str(movie)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for i in range(0, len(xy), stride):
            dots = sampled_dots(xy, phase, min(i + stride // 2, len(xy) - 1))
            frame = scope.draw(dots)

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
