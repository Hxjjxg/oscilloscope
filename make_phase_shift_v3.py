"""Compose and render Phase Shift v3 in the directory containing this script.

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
SIZE = 960
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
    phase = np.remainder(np.cumsum(hz(pitch)) * (TAU / RATE), TAU)
    return phase, gate, accent


# A drawing is one traversal per musical period. The 120 Hz geometry tables
# interpolate in both time and phase; geometry and timbre therefore move together.
TABLE = 512
TICKS = 120

def progress(t, start, end):
    return ease((t - start) / (end - start))


def boat(t, returning):
    if returning:
        sail = 1 - progress(t, 52.5, 57.5)
        position = .35 * (1 - progress(t, 50, 55))
        hull = 1 - progress(t, 56.0, 58.5)
    else:
        sail = .10 + .90 * progress(t, 1.25, 5.0)
        hull = .16 + .84 * progress(t, .12, 2.1)
        position = -.31 + .66 * progress(t, 6.25, 13.75)
    deck, mast = -.12, -.09
    head = deck + 1.14 * sail
    foot = deck + .08 * sail
    vertices = [(-1, deck), (-.66, deck-.36*hull), (.54, deck-.36*hull),
                (1, deck+.07*hull), (mast, deck), (mast, head),
                (-.82, foot), (mast, foot), (mast, head-.14*sail),
                (.67, foot), (mast+.08, foot), (mast, deck), (-1, deck)]
    points = np.array(vertices)
    angle = .065*np.sin(TAU*t/5) * sail
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    points = points @ rotation
    scale = .74
    if returning:
        scale *= 1 - progress(t, 57.5, 59.91)
    return points * scale + [position, -.09 + .043*np.sin(TAU*t/5)*sail]


def compass(t):
    # A needle attached to the rim at its tip provides a continuous beam path.
    u = t - 15
    angle = np.pi*(.5 - 1.24*progress(u, .4, 5.0) + .81*progress(u, 5.0, 8.75))
    angles = np.linspace(angle, angle+TAU, 145)
    # Four broad index marks survive the harmonic rounding at performance pitch.
    radii = .89 - .065*np.maximum(0,np.cos(4*angles))**18
    ring = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:,None]
    forward = np.array([np.cos(angle),np.sin(angle)])
    flank = np.array([-forward[1],forward[0]])
    along = [.17, .28, -.63, -.63, .28, .17, radii[0]]
    across = [.34, .087, .087, -.087, -.087, -.34, 0]
    needle = np.outer(along, forward) + np.outer(across, flank)
    points = np.vstack((ring, needle))
    tilt = .24 + .86*np.sin(np.pi*progress(u, 0, 10))
    points[:,0] *= np.cos(tilt)
    points[:,1] += .20*ring[0,0]*points[:,0]
    return points * (.80 + .10*progress(u, 0, 3))


def glider(t):
    # Coordinates in flight direction, wing span and height. The two wing
    # tips rotate around the longitudinal crease during the refrain.
    open_wings = .57 + .43*progress(t, 32.5, 37.5)
    # Bank through edge-on into a reversed silhouette on the peak A.
    bank = (.32 + 1.21*progress(t, 37.5, 41.25)
            + 1.30*progress(t, 41.25, 45.46875) - .65*progress(t, 46.25, 49.4))
    span = open_wings*np.array([0, 1, .13, 0, -.13, -1, 0, .13, 0, -.13, 0])
    height = np.array([0, .20, -.02, -.25, -.02, .20, 0, -.02, -.25, -.02, 0])
    length = np.array([1.12, -.87, -.59, -.48, -.59, -.87, 1.12, -.59, -.48, -.59, 1.12])
    vertical = span*np.cos(bank) + height*np.sin(bank)
    depth = height*np.cos(bank) - span*np.sin(bank)
    x = length + .34*span
    y = vertical*.81 + .23*length
    projection = 1/(1-.19*depth)
    scale = (.42 + .10*progress(t, 25, 32.5) + .36*progress(t, 32.5, 37.5))
    travel = progress(t, 25.6, 32.5)
    settle = progress(t, 32.5, 37.5)
    center = [-.59 + 1.10*travel - .53*settle,
              -.26 + .53*travel + .13*np.sin(np.pi*travel) - .28*settle]
    return np.column_stack((x,y))*projection[:,None]*scale + center


def waveform(t):
    if t < 15 or t >= 50:
        points = boat(t, t >= 50)
    elif t < 25:
        points = compass(t)
    else:
        points = glider(t)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0., np.cumsum(lengths)]
    where = np.arange(TABLE)*distance[-1]/TABLE
    wave = np.column_stack([np.interp(where,distance,axis) for axis in points.T])
    # Smooth corners with a periodic low-pass, retaining the silhouette while
    # suppressing the upper partials. The slow offsets place the craft in the frame.
    harmonics = np.arange(TABLE//2+1)
    taper = np.exp(-(harmonics/21)**4)
    return np.fft.irfft(np.fft.rfft(wave,axis=0)*taper[:,None],n=TABLE,axis=0)


def lead(t):
    phase, gate, accent = lead_controls(t)
    phase = phase * TABLE/TAU
    index = np.floor(phase).astype(int)
    fraction = (phase-index)[:,None]
    result = np.empty((len(t), 2), np.float32)
    block = RATE//TICKS
    current = waveform(0)
    for i in range(0,len(t),block):
        following = waveform((i+block)/RATE)
        end = min(i+block,len(t))
        key = index[i:end] % TABLE
        next_key = (key+1) % TABLE
        f = fraction[i:end]
        first = current[key] + (current[next_key]-current[key])*f
        last = following[key] + (following[next_key]-following[key])*f
        result[i:end] = first + (last-first)*(np.arange(end-i)/block)[:,None]
        current = following
    envelope = .72 + .27*gate + .03*accent
    return result * envelope[:,None]


def tone(mix, beat, length, note, gain, pan, attack, release, overtone):
    start = round(beat * BEAT * RATE)
    if start >= len(mix):
        return
    count = min(len(mix) - start, round(length * BEAT * RATE))
    t = np.arange(count) / RATE
    phase = TAU * hz(note) * t
    envelope = ease(t / attack) * ease((length * BEAT - t) / release)
    wave = (np.sin(phase) + overtone * np.sin(2 * phase + .3)) * (gain * envelope)
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
        if 2 <= bar < 10 or 14 <= bar < 20:
            for when, pitch, length, gain in ((0, root, 1.38, .072), (1.75, fifth, .48, .037),
                                              (2.5, root, 1.16, .057)):
                tone(mix, at + when, length, pitch, gain, .07, .018, .15, .23)
            percussion = .74 if 18 <= bar < 20 else 1
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


def master(path, duration):
    t = np.arange(round(duration * RATE)) / RATE
    xy = lead(t)
    accompaniment(xy)
    xy *= (ease(t / .041) * ease((len(t) / RATE - t) / .17))[:, None]
    peak = max(np.abs(signal.resample_poly(xy[max(0,i-32):i+RATE+32], 4, 1, axis=0)).max()
               for i in range(0,len(xy),RATE))
    xy *= 10 ** (-1.65 / 20) / peak
    sf.write(path, xy, RATE, subtype="PCM_24")


class Scope:
    """Render the beam on a 60 Hz display with a fast phosphor decay."""

    def __init__(self):
        self.exposure = np.zeros((SIZE, SIZE), np.uint8)
        self.stroke = np.zeros_like(self.exposure)
        self.persistence = np.zeros_like(self.exposure)

        ramp = np.arange(256) / 255
        self.palette = np.round(np.stack((239 * ramp ** .84, 246 * ramp ** .72,
                                         213 * ramp ** 1.51), axis=1)).astype(np.uint8)[:, None, :]

    def draw(self, xy):
        # Fixed-point coordinates preserve subpixel alignment of successive sweeps.
        points = np.rint((xy * [1, -1] * (SIZE * .54) + SIZE / 2) * 16).astype(np.int32)
        self.exposure.fill(0)
        intensity = round(100 * float(ease(np.ptp(xy, axis=0).max() / .033)))
        cuts = np.linspace(0, len(points) - 1, 5, dtype=int)
        if intensity:
            for first, last in zip(cuts[:-1], cuts[1:]):
                self.stroke.fill(0)
                cv2.polylines(self.stroke, [points[first:last + 1]], False, intensity, 2,
                              cv2.LINE_AA, shift=4)
                cv2.add(self.exposure, self.stroke, dst=self.exposure)

        cv2.addWeighted(self.persistence, .25, self.exposure, .75, 0, dst=self.persistence)
        halo = cv2.resize(self.persistence, (SIZE // 3, SIZE // 3), interpolation=cv2.INTER_AREA)
        halo = cv2.GaussianBlur(halo, (17, 17), 4.0)
        halo = cv2.resize(halo, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        texture = cv2.GaussianBlur(self.persistence, (9, 9), 1.9)
        light = cv2.addWeighted(self.persistence, .77, texture, .44, 0)
        light = cv2.addWeighted(light, 1, halo, .88, 0)
        return cv2.add(cv2.applyColorMap(light, self.palette), (13, 9, 7, 0))


def render(wav, movie):
    xy, rate = sf.read(wav, dtype="float32", always_2d=True)
    if rate != RATE or xy.shape[1] != 2 or len(xy) % (rate // FPS):
        raise ValueError("Expected frame-aligned 48 kHz stereo PCM")

    # The beam connects consecutive PCM samples in acquisition order.
    stride = rate // FPS
    scope = Scope()
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SIZE}x{SIZE}",
               "-framerate", str(FPS), "-i", "-", "-i", str(wav),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast",
               "-threads", "3", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "384k",
               "-movflags", "+faststart", "-metadata", f"title={TITLE}", str(movie)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for i in range(0, len(xy), stride):
            frame = scope.draw(xy[i:i + stride + 1])
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
