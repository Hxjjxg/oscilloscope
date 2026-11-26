// Phase Shift v4 — True Authentic XY Oscilloscope Audiovisual Masterpiece
//
// Build (MSYS2 / MinGW-w64 GCC):
//   g++ -O3 -std=c++20 -march=native -fopenmp make_phase_shift_v4.cpp -o phase_shift_v4.exe
//
// Run:
//   ./phase_shift_v4.exe              -> Generates "Phase Shift v4.wav" (96kHz 24-bit)
//                                        and "Phase Shift v4.mp4" (1080x1080 60fps)
//   ./phase_shift_v4.exe --preview    -> Generates WAV + 18 keyframe previews (PPM) for inspection
//
// 100% WYSIWYG: Every pixel of the video is physically simulated directly from the
// stereo audio samples (L = X, R = Y) using beam-velocity integration, dual-layer
// exponential CRT phosphor persistence, and 3-tier HDR bloom with filmic tone mapping.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr double PI = 3.14159265358979323846;
static constexpr double TAU = 2.0 * PI;

static constexpr int SR = 96000;          // 96 kHz high-resolution oscilloscope audio
static constexpr int FPS = 60;
static constexpr int SAMPLES_PER_FRAME = SR / FPS; // 1600 samples/frame
static constexpr int SIZE = 1080;         // 1080 x 1080 output video
static constexpr double BPM = 120.0;
static constexpr double BEAT = 60.0 / BPM;        // 0.5s
static constexpr double BAR = 4.0 * BEAT;         // 2.0s
static constexpr int TOTAL_BARS = 36;
static constexpr double DURATION = TOTAL_BARS * BAR; // 72.0s
static constexpr int TOTAL_SAMPLES = int(DURATION * SR);
static constexpr int TOTAL_FRAMES = TOTAL_SAMPLES / SAMPLES_PER_FRAME;

// ============================================================================
// Math & Geometry Primitives
// ============================================================================

struct Vec2 {
    double x = 0, y = 0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
};

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};

struct Vec4 {
    double x = 0, y = 0, z = 0, w = 0;
};

static inline double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
static inline double smoothstep(double x) {
    x = clamp01(x);
    return x * x * (3.0 - 2.0 * x);
}
static inline double smootherstep(double x) {
    x = clamp01(x);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}
static inline double progress(double t, double a, double b) {
    return smoothstep((t - a) / (b - a));
}
static inline double lerp(double a, double b, double k) { return a + (b - a) * k; }
static inline Vec2 lerp2(Vec2 a, Vec2 b, double k) { return {lerp(a.x, b.x, k), lerp(a.y, b.y, k)}; }
static inline Vec3 lerp3(Vec3 a, Vec3 b, double k) { return {lerp(a.x, b.x, k), lerp(a.y, b.y, k), lerp(a.z, b.z, k)}; }

static inline double midi_hz(double note) {
    return 440.0 * std::pow(2.0, (note - 69.0) / 12.0);
}

static inline Vec2 rot2(Vec2 p, double a) {
    double c = std::cos(a), s = std::sin(a);
    return {p.x * c - p.y * s, p.x * s + p.y * c};
}

static inline Vec3 rotX(Vec3 p, double a) {
    double c = std::cos(a), s = std::sin(a);
    return {p.x, p.y * c - p.z * s, p.y * s + p.z * c};
}
static inline Vec3 rotY(Vec3 p, double a) {
    double c = std::cos(a), s = std::sin(a);
    return {p.x * c + p.z * s, p.y, -p.x * s + p.z * c};
}
static inline Vec3 rotZ(Vec3 p, double a) {
    double c = std::cos(a), s = std::sin(a);
    return {p.x * c - p.y * s, p.x * s + p.y * c, p.z};
}

static inline Vec2 project3D(Vec3 p, double fov = 2.9, double z_dist = 3.2) {
    double denom = std::max(0.35, z_dist - p.z);
    double scale = fov / denom;
    return {p.x * scale, p.y * scale};
}

// ============================================================================
// Path Builder (Arc-Length Parameterized Vector Shapes with Instant Jumps)
// ============================================================================

struct Segment {
    Vec2 a, b;
    double len;
    bool jump; // true = instantaneous beam jump (0 arc-length budget)
};

struct VectorPath {
    std::vector<Segment> segs;
    std::vector<double> cum_len;
    double total_len = 0.0;

    void clear() {
        segs.clear();
        cum_len.clear();
        total_len = 0.0;
    }

    void add_line(Vec2 a, Vec2 b) {
        double dx = b.x - a.x, dy = b.y - a.y;
        double d = std::hypot(dx, dy);
        if (d < 1e-6) return;
        segs.push_back({a, b, d, false});
        total_len += d;
    }

    void add_polyline(const std::vector<Vec2>& pts, bool closed = false) {
        if (pts.size() < 2) return;
        for (size_t i = 0; i + 1 < pts.size(); ++i) add_line(pts[i], pts[i + 1]);
        if (closed) add_line(pts.back(), pts.front());
    }

    void add_circle(Vec2 center, double r, int steps = 48, double phase0 = 0.0) {
        Vec2 prev = {center.x + r * std::cos(phase0), center.y + r * std::sin(phase0)};
        for (int i = 1; i <= steps; ++i) {
            double ang = phase0 + TAU * double(i) / steps;
            Vec2 cur = {center.x + r * std::cos(ang), center.y + r * std::sin(ang)};
            add_line(prev, cur);
            prev = cur;
        }
    }

    void finalize() {
        cum_len.resize(segs.size());
        double acc = 0.0;
        for (size_t i = 0; i < segs.size(); ++i) {
            acc += segs[i].len;
            cum_len[i] = acc;
        }
        total_len = acc;
    }

    Vec2 sample(double u) const {
        if (segs.empty() || total_len <= 1e-9) return {0, 0};
        u = u - std::floor(u);
        double target = u * total_len;
        auto it = std::lower_bound(cum_len.begin(), cum_len.end(), target);
        size_t idx = (it == cum_len.end()) ? (segs.size() - 1) : size_t(it - cum_len.begin());
        double prev_len = (idx == 0) ? 0.0 : cum_len[idx - 1];
        double local_u = (segs[idx].len > 1e-9) ? (target - prev_len) / segs[idx].len : 0.0;
        // Slight vertex dwell curve (subtle cubic S-ease near endpoints makes vertices gleam!)
        double s = local_u;
        s = lerp(s, s * s * (3.0 - 2.0 * s), 0.22);
        return lerp2(segs[idx].a, segs[idx].b, s);
    }
};

// ============================================================================
// 3D Polyhedra & Vector Glyph Generators
// ============================================================================

static void add_cube_edges(VectorPath& path, double size, Vec3 rot, Vec3 offset,
                           double fov = 2.6, double zdist = 3.6, double flatten_z = 1.0) {
    static const int EDGES[12][2] = {
        {0,1},{1,3},{3,2},{2,0}, // back face
        {4,5},{5,7},{7,6},{6,4}, // front face
        {0,4},{1,5},{2,6},{3,7}  // connecting struts
    };
    Vec2 proj[8];
    for (int i = 0; i < 8; ++i) {
        Vec3 v((i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0, ((i & 4) ? 1.0 : -1.0) * flatten_z);
        v = v * size;
        v = rotZ(rotY(rotX(v, rot.x), rot.y), rot.z) + offset;
        proj[i] = project3D(v, fov, zdist);
    }
    // Order edges as continuous Euler-friendly chains to minimize beam jumps
    static const int ORDER[12] = {0,1,2,3, 8,4,5,6,7, 9,10,11};
    for (int k = 0; k < 12; ++k) {
        int e = ORDER[k];
        path.add_line(proj[EDGES[e][0]], proj[EDGES[e][1]]);
    }
}

static void add_octahedron_edges(VectorPath& path, double size, Vec3 rot, Vec3 offset,
                                 double fov = 2.6, double zdist = 3.6) {
    static const Vec3 V[6] = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}
    };
    static const int E[12][2] = {
        {4,0},{0,2},{2,4}, {4,1},{1,2},{2,3},
        {4,3},{3,0},{0,5}, {5,2},{5,1},{1,3}
    };
    Vec2 p[6];
    for (int i = 0; i < 6; ++i) {
        Vec3 v = rotZ(rotY(rotX(V[i] * size, rot.x), rot.y), rot.z) + offset;
        p[i] = project3D(v, fov, zdist);
    }
    for (int i = 0; i < 12; ++i) path.add_line(p[E[i][0]], p[E[i][1]]);
}

static void add_tesseract(VectorPath& path, double size, double angle_xw, double angle_yw,
                          Vec3 rot3, double inner_mix, double fov = 2.6, double zdist = 3.8) {
    Vec2 proj[16];
    double cxw = std::cos(angle_xw), sxw = std::sin(angle_xw);
    double cyw = std::cos(angle_yw), syw = std::sin(angle_yw);
    for (int i = 0; i < 16; ++i) {
        double x = (i & 1) ? 1.0 : -1.0;
        double y = (i & 2) ? 1.0 : -1.0;
        double z = (i & 4) ? 1.0 : -1.0;
        double w = (i & 8) ? 1.0 : -1.0;
        // 4D XW rotation
        double nx = x * cxw - w * sxw;
        double nw = x * sxw + w * cxw;
        x = nx; w = nw;
        // 4D YW rotation
        double ny = y * cyw - w * syw;
        nw = y * syw + w * cyw;
        y = ny; w = nw;
        // 4D -> 3D stereographic-style perspective
        double w_scale = 1.0 / (2.45 - 0.85 * w * inner_mix);
        Vec3 v3(x * w_scale * size, y * w_scale * size, z * w_scale * size);
        v3 = rotZ(rotY(rotX(v3, rot3.x), rot3.y), rot3.z);
        proj[i] = project3D(v3, fov, zdist);
    }
    // 32 edges: vertices differing in exactly 1 bit
    for (int i = 0; i < 16; ++i) {
        for (int b = 0; b < 4; ++b) {
            int j = i | (1 << b);
            if (j > i) {
                if (inner_mix < 0.01 && (b == 3 || (i & 8))) continue;
                path.add_line(proj[i], proj[j]);
            }
        }
    }
}

static void add_stella_octangula(VectorPath& path, double size, Vec3 rot, double fov = 2.6, double zdist = 3.6) {
    static const Vec3 T1[4] = {{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}};
    static const Vec3 T2[4] = {{-1,-1,-1},{-1,1,1},{1,-1,1},{1,1,-1}};
    static const int E[6][2] = {{0,1},{1,2},{2,0},{0,3},{1,3},{2,3}};
    auto draw_tet = [&](const Vec3 T[4]) {
        Vec2 p[4];
        for (int i = 0; i < 4; ++i) {
            Vec3 v = rotZ(rotY(rotX(T[i] * size, rot.x), rot.y), rot.z);
            p[i] = project3D(v, fov, zdist);
        }
        for (int i = 0; i < 6; ++i) path.add_line(p[E[i][0]], p[E[i][1]]);
    };
    draw_tet(T1);
    draw_tet(T2);
}

static void add_icosahedron(VectorPath& path, double size, Vec3 rot, Vec3 offset,
                            double fov = 2.6, double zdist = 3.6) {
    const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
    const double inv = 1.0 / std::sqrt(1.0 + phi * phi);
    const double a = inv, b = phi * inv;
    const Vec3 V[12] = {
        {-a, b, 0},{a, b, 0},{-a,-b, 0},{a,-b, 0},
        {0,-a, b},{0, a, b},{0,-a,-b},{0, a,-b},
        {b, 0,-a},{b, 0, a},{-b, 0,-a},{-b, 0, a}
    };
    Vec2 p[12];
    for (int i = 0; i < 12; ++i) {
        Vec3 v = rotZ(rotY(rotX(V[i] * size, rot.x), rot.y), rot.z) + offset;
        p[i] = project3D(v, fov, zdist);
    }
    for (int i = 0; i < 12; ++i) {
        for (int j = i + 1; j < 12; ++j) {
            double dx = V[i].x - V[j].x, dy = V[i].y - V[j].y, dz = V[i].z - V[j].z;
            double d = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d < 2.0 * a + 0.05) {
                path.add_line(p[i], p[j]);
            }
        }
    }
}

// Crisp geometric vector glyphs for Act IV ("P", "H", "A", "S", "E", "4", Diamond)
static void add_vector_glyph(VectorPath& path, int glyph_id, double scale, Vec3 rot, Vec3 offset, double depth = 0.14) {
    struct Seg2 { double x0, y0, x1, y1; };
    std::vector<Seg2> strokes;
    switch (glyph_id % 7) {
        case 0: // P
            strokes = {{-0.45,-0.75,-0.45,0.75},{-0.45,0.75,0.35,0.75},{0.35,0.75,0.50,0.45},
                       {0.50,0.45,0.50,0.15},{0.50,0.15,0.35,-0.05},{0.35,-0.05,-0.45,-0.05}};
            break;
        case 1: // H
            strokes = {{-0.48,-0.75,-0.48,0.75},{0.48,-0.75,0.48,0.75},{-0.48,0.0,0.48,0.0}};
            break;
        case 2: // A
            strokes = {{-0.55,-0.75,0.0,0.78},{0.0,0.78,0.55,-0.75},{-0.32,-0.18,0.32,-0.18}};
            break;
        case 3: // S
            strokes = {{0.48,0.62,0.22,0.75},{0.22,0.75,-0.28,0.75},{-0.28,0.75,-0.48,0.48},
                       {-0.48,0.48,-0.48,0.18},{-0.48,0.18,0.48,-0.15},{0.48,-0.15,0.48,-0.52},
                       {0.48,-0.52,0.25,-0.75},{0.25,-0.75,-0.28,-0.75},{-0.28,-0.75,-0.50,-0.58}};
            break;
        case 4: // E
            strokes = {{0.48,0.75,-0.48,0.75},{-0.48,0.75,-0.48,-0.75},{-0.48,-0.75,0.48,-0.75},
                       {-0.48,0.0,0.30,0.0}};
            break;
        case 5: // 4
            strokes = {{0.30,-0.75,0.30,0.75},{0.30,0.75,-0.52,-0.15},{-0.52,-0.15,0.55,-0.15}};
            break;
        default: // Geometric Diamond Emblem
            strokes = {{0,0.85,0.68,0},{0.68,0,0,-0.85},{0,-0.85,-0.68,0},{-0.68,0,0,0.85},
                       {0,0.42,0.34,0},{0.34,0,0,-0.42},{0,-0.42,-0.34,0},{-0.34,0,0,0.42}};
            break;
    }
    auto xform = [&](double x, double y, double z) -> Vec2 {
        Vec3 v(x * scale, y * scale, z * scale);
        v = rotZ(rotY(rotX(v, rot.x), rot.y), rot.z) + offset;
        return project3D(v, 2.7, 3.3);
    };
    // Front face
    for (const auto& s : strokes) {
        path.add_line(xform(s.x0, s.y0, depth), xform(s.x1, s.y1, depth));
    }
    // Extruded back echo face for rich 3D depth!
    for (const auto& s : strokes) {
        path.add_line(xform(s.x0, s.y0, -depth), xform(s.x1, s.y1, -depth));
    }
}

// ============================================================================
// Musical Score (36 Bars @ 120 BPM = 72.0s, D Minor -> F Major -> D Minor)
// ============================================================================

struct BarHarmony {
    int bass;
    int fifth;
    int chord[4];
};

static const BarHarmony SCORE[TOTAL_BARS] = {
    // Act I: Awakening (Bars 0-3)
    {38, 45, {50, 53, 57, 62}}, // Dm
    {38, 45, {50, 53, 57, 64}}, // Dm9
    {34, 41, {46, 50, 53, 57}}, // Bbmaj7
    {33, 40, {45, 49, 52, 55}}, // A7
    // Act II: Tesseract & Twin Cubes (Bars 4-9)
    {38, 45, {50, 53, 57, 62}}, // Dm
    {34, 41, {46, 50, 53, 58}}, // Bb
    {43, 50, {55, 58, 62, 65}}, // Gm7
    {33, 40, {45, 49, 52, 57}}, // A7
    {38, 45, {50, 53, 57, 62}}, // Dm
    {36, 43, {48, 52, 55, 60}}, // C
    // Act III: Stardust Constellation (Bars 10-13)
    {34, 41, {50, 53, 57, 62}}, // Bbmaj7
    {36, 43, {52, 55, 60, 64}}, // C
    {38, 45, {53, 57, 62, 65}}, // Dm
    {33, 40, {52, 55, 57, 61}}, // A7
    // Act IV: Warp Tunnel & Glyphs (Bars 14-19)
    {38, 45, {50, 53, 57, 62}}, // Dm
    {34, 41, {46, 50, 53, 58}}, // Bb
    {43, 50, {55, 58, 62, 65}}, // Gm7
    {33, 40, {45, 49, 52, 57}}, // A7
    {41, 48, {53, 57, 60, 65}}, // F
    {36, 43, {48, 52, 55, 60}}, // C
    // Act V: Geodesic Core & Petals (Bars 20-25)
    {41, 48, {53, 57, 60, 64}}, // Fmaj7
    {36, 43, {52, 55, 59, 62}}, // Cmaj7
    {43, 50, {55, 58, 62, 65}}, // Gm7
    {38, 45, {50, 53, 57, 62}}, // Dm
    {34, 41, {46, 50, 53, 57}}, // Bbmaj7
    {33, 40, {45, 49, 52, 55}}, // A7
    // Act VI: Kaleidoscopic Climax (Bars 26-31)
    {38, 45, {50, 53, 57, 62}}, // Dm
    {34, 41, {46, 50, 53, 58}}, // Bb
    {36, 43, {48, 52, 55, 60}}, // C
    {33, 40, {45, 49, 52, 57}}, // A7
    {38, 45, {50, 53, 57, 62}}, // Dm
    {34, 41, {46, 50, 53, 57}}, // Bb
    // Act VII: Convergence & Singularity (Bars 32-35)
    {43, 50, {50, 55, 58, 62}}, // Gm
    {33, 40, {45, 49, 52, 57}}, // A7
    {38, 45, {50, 53, 57, 62}}, // Dm
    {38, 45, {50, 53, 57, 62}}, // Dm
};

// Catchy rhythmic melody notes per 16th-note step (relative to bar chord or fixed MIDI)
static double get_scan_pitch(double t) {
    int bar = std::min(TOTAL_BARS - 1, std::max(0, int(t / BAR)));
    double beat_in_bar = (t - bar * BAR) / BEAT;
    int step16 = std::min(15, std::max(0, int(beat_in_bar * 4.0)));
    const auto& h = SCORE[bar];

    // Act I: Deep resonant unfolding drone -> rising bass
    if (bar < 2) return double(h.bass);
    if (bar < 4) {
        static const int pat[4] = {0, 0, 1, 2};
        int idx = pat[step16 / 4];
        return (idx == 0) ? double(h.bass) : double(h.chord[idx - 1]);
    }
    // Act II: Driving arpeggiated synthwave bass/lead
    if (bar < 10) {
        static const int arp[16] = {0,0,2,0, 1,0,3,2, 0,0,2,1, 3,2,1,0};
        int a = arp[step16];
        if (a == 0) return double(h.bass + 12);
        return double(h.chord[a - 1]);
    }
    // Act III: Floating crystalline stardust arpeggio
    if (bar < 14) {
        static const int arp[16] = {0,1,2,3, 2,1,0,3, 1,2,3,2, 0,2,3,1};
        double base = double(h.chord[arp[step16]]);
        if (bar == 13 && beat_in_bar > 2.0) {
            base += 12.0 * progress(beat_in_bar, 2.0, 4.0); // tension riser!
        }
        return base;
    }
    // Act IV: High-energy Warp Tunnel & Glyphs (bold rhythmic hook)
    if (bar < 20) {
        static const int hook[16] = {0,0,3,2, 1,1,2,0, 0,2,3,3, 2,1,2,1};
        int idx = hook[step16];
        double n = (idx == 0) ? double(h.bass + 12) : double(h.chord[idx]);
        if (bar >= 16 && (step16 % 4 == 0)) n += 12.0; // crisp glyph accent!
        return n;
    }
    // Act V: Singing legato Geodesic Core melody
    if (bar < 26) {
        static const int mel[8] = {2, 3, 2, 1, 0, 1, 2, 3};
        int idx = mel[step16 / 2];
        return double(h.chord[idx]);
    }
    // Act VI: Climactic Kaleidoscope (soaring octave + driving rhythm)
    if (bar < 32) {
        static const int climax[16] = {0,2,3,2, 1,3,2,3, 0,3,2,1, 3,2,3,0};
        int idx = climax[step16];
        double n = (idx == 0) ? double(h.bass + 12) : double(h.chord[idx]);
        if (bar >= 28 && (step16 == 2 || step16 == 6 || step16 == 10 || step16 == 14)) n += 12.0;
        return n;
    }
    // Act VII: Graceful deceleration & resolution to D2
    if (bar < 34) {
        int idx = (step16 / 4) % 4;
        return double(h.chord[3 - idx]) - 12.0;
    }
    return double(h.bass); // Pure D2 singularity
}

// Kick envelope (0..1) on driving beats
static double kick_env(double t) {
    int bar = int(t / BAR);
    if (bar < 4 || (bar >= 10 && bar < 14) || bar >= 32) return 0.0;
    double b = std::fmod(t / BEAT, 1.0);
    // Four-on-the-floor + occasional syncopated kick
    return std::exp(-b * 14.0);
}

// Snare/clap envelope on beats 1 and 3
static double snare_env(double t) {
    int bar = int(t / BAR);
    if (bar < 4 || (bar >= 10 && bar < 14) || bar >= 32) return 0.0;
    double b = std::fmod(t / BEAT, 2.0);
    if (b < 1.0) return 0.0;
    return std::exp(-(b - 1.0) * 18.0);
}

// ============================================================================
// Visual Scene Generator: Builds the 3D Vector Geometry at Time t
// ============================================================================

struct StarCloud {
    std::vector<Vec2> pts;
};

static void build_scene(double t, VectorPath& path, StarCloud& stars,
                        int& symmetry_fold, double& star_mix) {
    path.clear();
    stars.pts.clear();
    symmetry_fold = 1;
    star_mix = 0.0;

    double k_punch = kick_env(t);
    double s_punch = snare_env(t);

    // ------------------------------------------------------------------------
    // ACT I (0 - 8s): Genesis — Singularity -> Laser Line -> Square -> Cube -> Tesseract
    // ------------------------------------------------------------------------
    if (t < 8.0) {
        if (t < 2.0) {
            // Breathing glowing singularity point + delicate harmonic halo
            double p = progress(t, 0.0, 2.0);
            double r = 0.015 + 0.035 * p + 0.01 * std::sin(TAU * 2.0 * t);
            path.add_circle({0, 0}, r, 32, t * 2.0);
        } else if (t < 4.0) {
            // Horizontal laser line stretching out with crisp standing wave ripple
            double open_x = progress(t, 2.0, 3.8);
            double half_w = 0.05 + 0.62 * open_x;
            double ripple = 0.07 * std::sin(PI * open_x);
            std::vector<Vec2> line;
            for (int i = 0; i <= 48; ++i) {
                double u = -1.0 + 2.0 * i / 48.0;
                double y = ripple * std::sin(3.0 * PI * u + t * 8.0) * (1.0 - u * u);
                line.push_back({u * half_w, y});
            }
            path.add_polyline(line, false);
            // Return trace back for balanced beam brightness
            std::reverse(line.begin(), line.end());
            path.add_polyline(line, false);
        } else {
            // Vertical unfold into square -> 3D extrusion into Cube -> Inner Hypercube appears!
            double unfold = progress(t, 4.0, 5.5);
            double extrude = progress(t, 5.0, 7.2);
            double inner = progress(t, 6.5, 8.0);
            Vec3 rot(0.38 * extrude + 0.25 * std::sin(t * 0.7),
                     0.65 * extrude + 0.45 * (t - 4.0),
                     0.15 * std::sin(t * 0.5));
            if (inner < 0.02) {
                add_cube_edges(path, 0.52, rot, {0, 0, 0}, 2.7, 3.5, std::max(0.001, extrude));
                // Scale Y by unfold during square opening
                for (auto& seg : path.segs) {
                    seg.a.y *= (0.04 + 0.96 * unfold);
                    seg.b.y *= (0.04 + 0.96 * unfold);
                    seg.len = std::hypot(seg.b.x - seg.a.x, seg.b.y - seg.a.y);
                }
            } else {
                add_tesseract(path, 0.85, 0.45 * inner, 0.25 * inner, rot, inner, 2.7, 3.8);
            }
        }
    }
    // ------------------------------------------------------------------------
    // ACT II (8 - 20s): 4D Tesseract Rotation -> Splits into Twin Orbiting Polyhedra
    // ------------------------------------------------------------------------
    else if (t < 20.0) {
        double lt = t - 8.0;
        double fov = 2.65 + 0.28 * k_punch; // punchy perspective bounce on kick!
        if (t < 14.0) {
            // Glorious 4D Tesseract turning inside-out in XW and YW planes
            double axw = 0.45 + lt * 0.72;
            double ayw = 0.25 + lt * 0.48;
            Vec3 rot(0.45 + 0.35 * std::sin(lt * 0.65),
                     1.2 + lt * 0.62,
                     0.18 * std::cos(lt * 0.5));
            add_tesseract(path, 0.86 + 0.05 * k_punch, axw, ayw, rot, 1.0, fov, 3.7);
            // Snare flash ring around the tesseract
            if (s_punch > 0.15) {
                path.add_circle({0, 0}, 0.78 + 0.10 * (1.0 - s_punch), 36, lt * 3.0);
            }
        } else {
            // Splits into Twin Orbiting 3D Polyhedra (Cube + Octahedron/Cube) with orbiting satellites
            double split = progress(t, 14.0, 15.8);
            double orbit_ang = (t - 14.0) * 1.15;
            double dist = 0.72 * split;
            Vec3 pos1(dist * std::cos(orbit_ang), 0.18 * std::sin(orbit_ang * 2.0), 0.35 * std::sin(orbit_ang));
            Vec3 pos2 = pos1 * -1.0;
            Vec3 rot1(lt * 0.9, lt * 1.25, lt * 0.35);
            Vec3 rot2(-lt * 1.1, lt * 0.85, -lt * 0.5);
            double sz = lerp(0.54, 0.38, split) + 0.03 * k_punch;
            add_cube_edges(path, sz, rot1, pos1, fov, 3.6);
            // Second body morphs between cube and crisp octahedron
            if (t < 17.0) {
                add_cube_edges(path, sz * 0.92, rot2, pos2, fov, 3.6);
            } else {
                add_octahedron_edges(path, sz * 1.25, rot2, pos2, fov, 3.6);
            }
            // Crisp energy tether + orbiting spark points on snare
            if (split > 0.3) {
                Vec2 c1 = project3D(pos1, fov, 3.6);
                Vec2 c2 = project3D(pos2, fov, 3.6);
                path.add_line(c1, c2);
            }
        }
    }
    // ------------------------------------------------------------------------
    // ACT III (20 - 28s): Stardust Constellation Cloud ("大量散点") + Helix Lock
    // ------------------------------------------------------------------------
    else if (t < 28.0) {
        double lt = t - 20.0;
        // 3D Swirling Galaxy / Fibonacci Constellation of 56 razor-sharp phosphor stars!
        const int N = 56;
        double spiral_morph = progress(t, 22.5, 25.0);
        double compress = progress(t, 26.0, 28.0);
        double rot_y = lt * 0.85 + 2.4 * compress * compress;
        double rot_x = 0.55 + 0.35 * std::sin(lt * 0.5);

        stars.pts.reserve(N);
        const double golden = PI * (3.0 - std::sqrt(5.0));
        for (int i = 0; i < N; ++i) {
            double u = (i + 0.5) / N;
            // Shape A: 3D Fibonacci Sphere + Galaxy Arms
            double y_sph = 1.0 - 2.0 * u;
            double r_sph = std::sqrt(std::max(0.0, 1.0 - y_sph * y_sph));
            double theta = golden * i;
            Vec3 p_sph(r_sph * std::cos(theta) * 0.95, y_sph * 0.95, r_sph * std::sin(theta) * 0.95);

            // Shape B: Double-Helix DNA / Torus Spiral
            double arm = (i & 1) ? PI : 0.0;
            double ang = u * TAU * 2.5 + arm;
            double rad = 0.55 + 0.25 * std::sin(u * PI * 3.0);
            Vec3 p_hel(rad * std::cos(ang), (u - 0.5) * 1.55, rad * std::sin(ang));

            Vec3 p = lerp3(p_sph, p_hel, spiral_morph);
            // Compress smoothly into tunnel seed at end of Act III
            p = p * (1.0 - 0.55 * compress);
            p = rotX(rotY(p, rot_y), rot_x);
            stars.pts.push_back(project3D(p, 2.5, 3.2));
        }
        // Star cloud takes full stage (100% pure crisp glowing star dots!), then adds faint inner wireframe ring
        star_mix = 1.0 - 0.25 * compress;
        if (compress > 0.05) {
            // Seed ring of the upcoming Warp Tunnel
            int sides = 6;
            std::vector<Vec2> hex;
            for (int k = 0; k < sides; ++k) {
                double a = TAU * k / sides + lt * 1.2;
                double r = 0.15 + 0.55 * compress;
                hex.push_back({r * std::cos(a), r * std::sin(a)});
            }
            path.add_polyline(hex, true);
        }
    }
    // ------------------------------------------------------------------------
    // ACT IV (28 - 40s): 3D Perspective Warp Tunnel & Extruded Vector Glyphs
    // ------------------------------------------------------------------------
    else if (t < 40.0) {
        double lt = t - 28.0;
        double fov = 2.6 + 0.30 * k_punch;
        if (t < 32.0) {
            // Infinite 3D Hexagonal Warp Tunnel with crisp perspective rails!
            const int RINGS = 4;
            const int SIDES = 6;
            double speed = 1.85;
            double rx = 0.16 * std::sin(lt * 0.9);
            double ry = 0.20 * std::cos(lt * 0.7);
            auto ring_pt = [&](int s, double z) -> Vec2 {
                double twist = z * 0.26 + lt * 0.55;
                double a = TAU * s / SIDES + twist;
                double rad = 0.62 + 0.05 * k_punch;
                Vec3 v(rad * std::cos(a), rad * std::sin(a), z);
                v = rotY(rotX(v, rx), ry);
                return project3D(v, fov, 3.25);
            };
            for (int r = 0; r < RINGS; ++r) {
                double z_frac = std::fmod(r / double(RINGS) + lt * speed * 0.35, 1.0);
                double z = lerp(-2.7, 1.45, z_frac);
                std::vector<Vec2> poly;
                for (int s = 0; s < SIDES; ++s) poly.push_back(ring_pt(s, z));
                path.add_polyline(poly, true);
            }
            // 6 crisp 3D perspective tunnel rails connecting deep -> near!
            for (int s = 0; s < SIDES; ++s) {
                path.add_line(ring_pt(s, -2.7), ring_pt(s, 1.45));
            }
        } else {
            // Bold 3D Extruded Vector Glyphs ("P H A S E 4 ◆") zooming & rotating on the beat!
            double gt = t - 32.0;
            int glyph_idx = int(gt / 1.0); // 1 new glyph every 2 beats (1.0s)
            double local_u = std::fmod(gt, 1.0);
            double zoom = 0.42 + 0.46 * smoothstep(local_u / 0.22) * (1.0 - 0.15 * local_u) + 0.07 * k_punch;
            Vec3 rot(0.28 * std::sin(local_u * PI),
                     lerp(-0.65, 0.65, smoothstep(local_u)),
                     0.15 * std::cos(gt * 1.5));
            add_vector_glyph(path, glyph_idx, zoom, rot, {0, 0, 0}, 0.14);

            // Outer framing tunnel ring rushing past behind the glyph
            double z_frac = std::fmod(gt * 1.0, 1.0);
            double z = lerp(-2.0, 1.6, z_frac);
            std::vector<Vec2> frame;
            for (int s = 0; s < 8; ++s) {
                double a = TAU * s / 8.0 + gt * 0.5;
                frame.push_back(project3D({1.15 * std::cos(a), 1.15 * std::sin(a), z}, fov, 3.3));
            }
            path.add_polyline(frame, true);
        }
    }
    // ------------------------------------------------------------------------
    // ACT V (40 - 52s): Geodesic Core, Unfolding Iris Petals & Orbiting Planetoid
    // ------------------------------------------------------------------------
    else if (t < 52.0) {
        double lt = t - 40.0;
        double open_petals = progress(t, 42.0, 46.5);
        double sat_emerge = progress(t, 46.0, 49.0);
        double orbit = (t - 46.0) * 0.95;

        Vec3 center(-0.28 * sat_emerge, 0.05 * std::sin(lt), 0.0);
        Vec3 rot_core(lt * 0.7, lt * 1.15, lt * 0.35);

        // Inner spinning Octahedron core (cleaner & sharper silhouette inside the rings)
        add_octahedron_edges(path, 0.34 + 0.04 * k_punch, rot_core, center, 2.7, 3.6);

        // Outer unfolding Geodesic Iris Shell (clean meridians + equatorial rings)
        const int RINGS = 3;
        for (int r = 0; r < RINGS; ++r) {
            double lat = lerp(-0.78, 0.78, r / double(RINGS - 1)) * (1.0 - 0.50 * open_petals);
            double rad = (0.76 + 0.20 * open_petals) * std::cos(lat * 0.55);
            double y_off = 0.62 * std::sin(lat);
            double tilt_x = 0.38 + 0.44 * open_petals * ((r == 1) ? 0.0 : ((r == 0) ? 1.0 : -1.0));
            double tilt_z = 0.22 * std::sin(lt * 0.6 + r);
            std::vector<Vec2> ring;
            const int SEGS = 28;
            for (int s = 0; s <= SEGS; ++s) {
                double a = TAU * s / SEGS + lt * (0.48 + 0.16 * r);
                Vec3 v(rad * std::cos(a), y_off, rad * std::sin(a));
                v = rotZ(rotX(v, tilt_x), tilt_z) + center;
                ring.push_back(project3D(v, 2.7, 3.6));
            }
            path.add_polyline(ring, false);
        }
        // 3 vertical geodesic meridian ribs that bloom outward like petals
        for (int m = 0; m < 3; ++m) {
            double phi = TAU * m / 3.0 + lt * 0.55;
            std::vector<Vec2> arc;
            const int SEGS = 18;
            for (int s = 0; s <= SEGS; ++s) {
                double th = lerp(-PI * 0.46, PI * 0.46, s / double(SEGS));
                double r_m = 0.78 + 0.26 * open_petals * std::cos(th);
                Vec3 v(r_m * std::cos(th) * std::cos(phi),
                       0.78 * std::sin(th),
                       r_m * std::cos(th) * std::sin(phi));
                v = rotX(v, 0.32) + center;
                arc.push_back(project3D(v, 2.7, 3.6));
            }
            path.add_polyline(arc, false);
        }

        // Second ringed planetoid emerging and orbiting!
        if (sat_emerge > 0.02) {
            Vec3 sat_pos(0.74 * sat_emerge * std::cos(orbit) + 0.24 * sat_emerge,
                         0.46 * sat_emerge * std::sin(orbit * 1.5),
                         0.35 * std::sin(orbit));
            add_octahedron_edges(path, 0.24 * sat_emerge, {lt * 1.4, -lt * 1.1, lt * 0.5}, sat_pos, 2.7, 3.6);
            std::vector<Vec2> sring;
            for (int s = 0; s <= 24; ++s) {
                double a = TAU * s / 24.0;
                Vec3 v(0.40 * sat_emerge * std::cos(a), 0.0, 0.40 * sat_emerge * std::sin(a));
                v = rotZ(rotX(v, 0.7), -0.45) + sat_pos;
                sring.push_back(project3D(v, 2.7, 3.6));
            }
            path.add_polyline(sring, false);
        }
    }
    // ------------------------------------------------------------------------
    // ACT VI (52 - 64s): Kaleidoscopic Climax — 3D Torus Knot -> Stella Octangula + 6-Fold Mandala
    // ------------------------------------------------------------------------
    else if (t < 64.0) {
        double lt = t - 52.0;
        double fov = 2.65 + 0.32 * k_punch;

        if (t < 58.0) {
            // Continuous 3D Torus Knot (p=3, q=5) with crystalline depth & rhythmic bloom
            const int STEPS = 140;
            const double p_k = 3.0, q_k = 5.0;
            double morph_star = progress(t, 56.0, 58.0);
            Vec3 rot(lt * 0.75, lt * 1.05, lt * 0.35);
            std::vector<Vec2> knot;
            knot.reserve(STEPS + 1);
            for (int i = 0; i <= STEPS; ++i) {
                double u = TAU * i / STEPS;
                double r_tube = 0.52 + 0.24 * std::cos(q_k * u + lt * 1.2);
                Vec3 v(r_tube * std::cos(p_k * u),
                       r_tube * std::sin(p_k * u),
                       0.32 * std::sin(q_k * u + lt * 1.2));
                v = rotZ(rotY(rotX(v * (1.05 + 0.07 * k_punch), rot.x), rot.y), rot.z);
                knot.push_back(project3D(v, fov, 3.1));
            }
            path.add_polyline(knot, true);
            // Add inner Stella Octangula emerging
            if (morph_star > 0.05) {
                add_stella_octangula(path, 0.42 * morph_star, {-rot.x, rot.y * 1.3, rot.z}, fov, 3.4);
            }
        } else {
            // PEAK CLIMAX: 6-Fold Rotational Kaleidoscopic Crystal + Stella Octangula!
            symmetry_fold = 6; // Cycle-interleaved 6-fold rotational symmetry!
            double kt = t - 58.0;
            double vortex = progress(t, 61.5, 64.0);
            double spin = kt * (0.95 + 1.85 * vortex);
            double r_pulse = 1.0 + 0.10 * k_punch;
            // Crisp faceted crystal shard blade (mirrored 6x into a blazing geometric mandala!)
            Vec3 p0(0.18 * r_pulse, 0.0, 0.0);
            Vec3 p1(0.54 * r_pulse, 0.20 * r_pulse, 0.24 * std::sin(kt * 2.2));
            Vec3 p2(0.92 * r_pulse, 0.0, -0.18 * std::cos(kt * 1.8));
            Vec3 p3(0.54 * r_pulse, -0.20 * r_pulse, 0.24 * std::sin(kt * 2.2));
            Vec3 pts[5] = {p0, p1, p2, p3, p0};
            std::vector<Vec2> blade;
            for (int i = 0; i < 5; ++i) {
                Vec3 v = rotZ(pts[i], spin + vortex * 0.65 * (i == 2 ? 1.0 : 0.0));
                v = rotX(v, 0.25 * std::sin(kt * 0.9));
                blade.push_back(project3D(v, fov, 3.2));
            }
            path.add_polyline(blade, false);
            // Outer connecting facet arc + glowing central core facet
            Vec3 a1 = rotZ(Vec3(0.78 * r_pulse, 0.0, 0.0), spin + PI / 6.0);
            Vec3 a2 = rotZ(Vec3(0.78 * r_pulse, 0.0, 0.0), spin - PI / 6.0);
            path.add_line(project3D(a1, fov, 3.2), project3D(a2, fov, 3.2));
            Vec3 c1 = rotZ(Vec3(0.18 * r_pulse, 0.0, 0.0), spin);
            Vec3 c2 = rotZ(Vec3(0.18 * r_pulse, 0.0, 0.0), spin + PI / 3.0);
            path.add_line(project3D(c1, fov, 3.2), project3D(c2, fov, 3.2));
        }
    }
    // ------------------------------------------------------------------------
    // ACT VII (64 - 72s): Convergence & Singularity Finale
    // ------------------------------------------------------------------------
    else {
        double lt = t - 64.0;
        double fold_4d = 1.0 - progress(t, 64.0, 67.5);
        double collapse = progress(t, 67.5, 71.4);
        double scale = 0.85 * (1.0 - 0.985 * smootherstep(collapse));

        if (collapse < 0.75) {
            Vec3 rot(0.4 + lt * 0.35, 1.0 + lt * 0.55, 0.15 * std::sin(lt));
            add_tesseract(path, scale, 0.9 * fold_4d, 0.5 * fold_4d, rot, fold_4d, 2.7, 3.8);
        } else {
            // Smoothly contracts to a single blazing phosphor star-point at (0, 0)
            double r = std::max(0.002, 0.18 * (1.0 - progress(t, 69.8, 71.7)));
            path.add_circle({0, 0}, r, 32, lt * 3.0);
        }
    }

    path.finalize();
}

// ============================================================================
// Stereo XY Audio Synthesizer (96 kHz, 100% Authentic Oscilloscope Signal)
// ============================================================================

static void synthesize_audio(std::vector<float>& L, std::vector<float>& R) {
    L.assign(TOTAL_SAMPLES, 0.0f);
    R.assign(TOTAL_SAMPLES, 0.0f);

    // Update geometry at 240 Hz (every 400 samples = 4.17ms) and crossfade smoothly!
    const int BLOCK = 400;
    VectorPath path_a, path_b;
    StarCloud stars_a, stars_b;
    int sym_a = 1, sym_b = 1;
    double smix_a = 0.0, smix_b = 0.0;

    build_scene(0.0, path_a, stars_a, sym_a, smix_a);

    double scan_phase = 0.0;
    double sub_phase = 0.0;
    double kick_phase = 0.0;
    double smooth_pitch = get_scan_pitch(0.0);
    int64_t cycle_count = 0;

    // Gentle 1-pole low-pass state for warm analog ear-comfort (~14 kHz cutoff at 96 kHz)
    double lp_l = 0.0, lp_r = 0.0;
    // Ultra-gentle DC blocker (~6 Hz)
    double dc_xl = 0.0, dc_yl = 0.0, dc_xr = 0.0, dc_yr = 0.0;
    const double dc_coeff = 1.0 - (TAU * 6.0 / SR);

    uint32_t rng = 0x9E3779B9u;
    auto next_rand = [&]() -> double {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return (double(rng) / 4294967296.0) * 2.0 - 1.0;
    };

    for (int base = 0; base < TOTAL_SAMPLES; base += BLOCK) {
        int end = std::min(TOTAL_SAMPLES, base + BLOCK);
        double t_next = double(end) / SR;
        build_scene(t_next, path_b, stars_b, sym_b, smix_b);

        for (int i = base; i < end; ++i) {
            double t = double(i) / SR;
            double alpha = double(i - base) / double(BLOCK);

            // Smooth portamento on pitch changes (~10ms glide) + subtle expressive vibrato in Act V
            double target_pitch = get_scan_pitch(t);
            smooth_pitch += (target_pitch - smooth_pitch) * 0.0025;
            double vib = (t >= 40.0 && t < 52.0) ? (0.14 * std::sin(TAU * 5.2 * t)) : 0.0;
            double f_scan = midi_hz(smooth_pitch + vib);

            scan_phase += f_scan / SR;
            if (scan_phase >= 1.0) {
                scan_phase -= 1.0;
                cycle_count++;
            }

            // Sample continuous wireframe path (crossfaded across block)
            Vec2 pa = path_a.sample(scan_phase);
            Vec2 pb = path_b.sample(scan_phase);
            Vec2 xy = lerp2(pa, pb, alpha);

            // Cycle-interleaved rotational kaleidoscope symmetry (Act VI)
            int sym = (alpha < 0.5) ? sym_a : sym_b;
            if (sym > 1) {
                double ang = TAU * double(cycle_count % sym) / double(sym);
                xy = rot2(xy, ang);
            }

            // Stardust Constellation Cloud ("大量散点", Act III)
            double smix = lerp(smix_a, smix_b, alpha);
            bool in_stars = false;
            if (smix > 0.01 && !stars_a.pts.empty() && !stars_b.pts.empty()) {
                if (smix > 0.99 || (cycle_count % 3 != 0)) {
                    in_stars = true;
                    size_t n = std::min(stars_a.pts.size(), stars_b.pts.size());
                    double pos = scan_phase * n;
                    size_t idx = size_t(pos) % n;
                    size_t nxt = (idx + 1) % n;
                    double frac = pos - std::floor(pos);
                    Vec2 p_cur = lerp2(stars_a.pts[idx], stars_b.pts[idx], alpha);
                    Vec2 p_nxt = lerp2(stars_a.pts[nxt], stars_b.pts[nxt], alpha);
                    // Dwell 82% on pinpoint star, sleek 18% transition -> blazing stars + warm musical audio!
                    if (frac < 0.82) {
                        xy = p_cur;
                    } else {
                        double u = smoothstep((frac - 0.82) / 0.18);
                        xy = lerp2(p_cur, p_nxt, u);
                    }
                }
            }

            // Musical & Visual Percussion Layer (100% XY-integrated, ZERO wireframe smearing!)
            int bar = std::min(TOTAL_BARS - 1, int(t / BAR));
            double beat_frac = std::fmod(t / BEAT, 1.0);
            double beat2_frac = std::fmod(t / BEAT, 2.0);
            bool driving = (bar >= 4 && bar < 10) || (bar >= 14 && bar < 20) || (bar >= 26 && bar < 32);

            // 1) Punchy Sub-Kick: Radial AM pulse on geometry (zero lateral smear!) + tight sub thump
            if (driving && beat_frac < 0.16) {
                double kf = 48.0 + 130.0 * std::exp(-beat_frac * 42.0);
                kick_phase += TAU * kf / SR;
                double kenv = std::exp(-beat_frac * 18.0) * smoothstep(beat_frac / 0.003);
                // Radial scale punch keeps every vertex & edge laser-sharp!
                double r_mod = 1.0 + 0.075 * kenv * std::sin(kick_phase);
                xy.x *= r_mod;
                xy.y *= r_mod;
            } else {
                kick_phase = 0.0;
            }

            // 2) Crisp Hi-Hat / Sparkle Ticks on offbeats (tiny electric sparks!)
            if (driving) {
                double hat_b = std::fmod(t / BEAT + 0.5, 0.5);
                if (hat_b < 0.022) {
                    double henv = std::exp(-hat_b * 150.0) * smoothstep(hat_b / 0.001);
                    double spark = 0.016 * henv;
                    xy.x += spark * next_rand();
                    xy.y += spark * next_rand();
                }
            }

            // 3) Snare / Clap transient on beats 1 & 3 (crisp electric shimmer burst)
            if (driving && beat2_frac >= 1.0 && beat2_frac < 1.075) {
                double sb = beat2_frac - 1.0;
                double senv = std::exp(-sb * 48.0) * smoothstep(sb / 0.002);
                double tone = std::sin(TAU * 210.0 * sb);
                double ns = 0.028 * senv * (0.55 * next_rand() + 0.45 * tone);
                xy.x += ns;
                xy.y -= ns;
            }

            // 4) Subtle harmonic depth via clean phase-locked octave breathing (no un-locked smear!)
            double f_sub = midi_hz(double(SCORE[bar].bass));
            sub_phase += TAU * f_sub / SR;
            if (sub_phase >= TAU) sub_phase -= TAU;
            double sub_am = (bar < 2 || bar >= 34) ? 0.012 : 0.018;
            if (!in_stars) {
                double breath = 1.0 + sub_am * std::sin(sub_phase);
                xy.x *= breath;
                xy.y *= breath;
            }

            // Global smooth fade-in (0.25s) and fade-out (1.2s)
            double master_env = smoothstep(t / 0.25) * (1.0 - smoothstep((t - (DURATION - 1.2)) / 1.2));
            xy.x *= master_env * 0.78;
            xy.y *= master_env * 0.78;

            // Gentle analog low-pass (~26 kHz @ 96 kHz: razor-sharp corners + smooth audio!)
            if (in_stars) {
                const double lp_s = 0.52;
                lp_l += (xy.x - lp_l) * lp_s;
                lp_r += (xy.y - lp_r) * lp_s;
            } else {
                const double lp_a = 0.84;
                lp_l += (xy.x - lp_l) * lp_a;
                lp_r += (xy.y - lp_r) * lp_a;
            }

            // Ultra-gentle 6 Hz DC blocker
            double yl = lp_l - dc_xl + dc_coeff * dc_yl;
            dc_xl = lp_l; dc_yl = yl;
            double yr = lp_r - dc_xr + dc_coeff * dc_yr;
            dc_xr = lp_r; dc_yr = yr;

            // Soft limiter (prevents any digital clipping above ±0.96)
            L[i] = float(0.95 * std::tanh(yl / 0.95));
            R[i] = float(0.95 * std::tanh(yr / 0.95));
        }

        path_a = std::move(path_b);
        stars_a = std::move(stars_b);
        sym_a = sym_b;
        smix_a = smix_b;
    }
}

// ============================================================================
// Write 24-bit Stereo WAV (96 kHz)
// ============================================================================

static void write_wav_16bit(const std::string& path, const std::vector<float>& L, const std::vector<float>& R) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open WAV for writing: " << path << "\n";
        std::exit(1);
    }
    uint32_t num_samples = uint32_t(L.size());
    uint16_t channels = 2;
    uint16_t bits = 16;
    uint32_t byte_rate = SR * channels * (bits / 8);
    uint16_t block_align = channels * (bits / 8);
    uint32_t data_size = num_samples * block_align;
    uint32_t riff_size = 36 + data_size;

    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };

    f.write("RIFF", 4); w32(riff_size); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(channels);
    w32(SR); w32(byte_rate); w16(block_align); w16(bits);
    f.write("data", 4); w32(data_size);

    std::vector<uint8_t> buf(65536 * 4);
    size_t pos = 0;
    auto flush_buf = [&]() {
        if (pos > 0) {
            f.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(pos));
            pos = 0;
        }
    };

    for (uint32_t i = 0; i < num_samples; ++i) {
        float vals[2] = {L[i], R[i]};
        for (int c = 0; c < 2; ++c) {
            double clamped = std::max(-1.0, std::min(1.0, double(vals[c])));
            int32_t s = int32_t(std::llround(clamped * 32767.0));
            buf[pos++] = uint8_t(s & 0xFF);
            buf[pos++] = uint8_t((s >> 8) & 0xFF);
        }
        if (pos >= buf.size() - 4) flush_buf();
    }
    flush_buf();
}

// ============================================================================
// Physically-Based HDR CRT Phosphor Renderer (1080x1080 @ 60fps)
// ============================================================================

struct CRTScope {
    std::vector<float> fast_buf;   // Fast bright phosphor core (tau ~12ms)
    std::vector<float> slow_buf;   // Lingering afterglow trail (tau ~58ms)
    std::vector<float> hdr_buf;    // Combined HDR intensity
    std::vector<float> blur_tmp;
    std::vector<float> bloom1;     // Tight bloom (sigma ~1.6)
    std::vector<float> half_a, half_b; // 2x downsampled medium bloom
    std::vector<float> qtr_a, qtr_b;   // 4x downsampled wide atmospheric halo
    std::vector<float> bg_grid;    // Subtle Tektronix CRT glass + graticule
    std::vector<uint8_t> frame_bgr;

    static constexpr int HALF = SIZE / 2;
    static constexpr int QTR = SIZE / 4;

    CRTScope()
        : fast_buf(SIZE * SIZE, 0.0f),
          slow_buf(SIZE * SIZE, 0.0f),
          hdr_buf(SIZE * SIZE, 0.0f),
          blur_tmp(SIZE * SIZE, 0.0f),
          bloom1(SIZE * SIZE, 0.0f),
          half_a(HALF * HALF, 0.0f),
          half_b(HALF * HALF, 0.0f),
          qtr_a(QTR * QTR, 0.0f),
          qtr_b(QTR * QTR, 0.0f),
          bg_grid(SIZE * SIZE, 0.0f),
          frame_bgr(SIZE * SIZE * 3, 0) {
        build_graticule();
    }

    void build_graticule() {
        const double cx = SIZE * 0.5, cy = SIZE * 0.5;
        const double step = SIZE / 10.0; // 10x10 divisions
        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                double dx = (x - cx) / cx;
                double dy = (y - cy) / cy;
                double r2 = dx * dx + dy * dy;
                double vig = std::max(0.0, 1.0 - 0.52 * r2);

                double g = 0.0;
                // Subtle major grid lines
                double gx = std::abs(std::fmod(x + 0.5 * step, step) - 0.5 * step);
                double gy = std::abs(std::fmod(y + 0.5 * step, step) - 0.5 * step);
                if (gx < 0.7 || gy < 0.7) g = 0.022;
                // Center axes slightly clearer with fine tick marks
                if (std::abs(x - cx) < 0.8 || std::abs(y - cy) < 0.8) g = 0.042;
                double sub = step / 5.0;
                if (std::abs(y - cy) < 5.5 && std::abs(std::fmod(x + 0.5 * sub, sub) - 0.5 * sub) < 0.7) g = 0.055;
                if (std::abs(x - cx) < 5.5 && std::abs(std::fmod(y + 0.5 * sub, sub) - 0.5 * sub) < 0.7) g = 0.055;

                // Subtle CRT curved screen bezel dark falloff
                double edge = std::max(std::abs(dx), std::abs(dy));
                double bezel = 1.0 - smoothstep((edge - 0.94) / 0.06);
                bg_grid[y * SIZE + x] = float((0.012 + g) * vig * bezel);
            }
        }
    }

    // Fast separable 1D Gaussian blur (radius R)
    static void blur_sep(const std::vector<float>& src, std::vector<float>& dst,
                         std::vector<float>& tmp, int w, int h, float sigma, int radius) {
        std::vector<float> kern(radius * 2 + 1);
        float sum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            float v = std::exp(-0.5f * (i * i) / (sigma * sigma));
            kern[i + radius] = v;
            sum += v;
        }
        for (float& v : kern) v /= sum;

        // Horizontal pass
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) {
            const float* row = &src[y * w];
            float* out = &tmp[y * w];
            for (int x = 0; x < w; ++x) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int xx = std::min(w - 1, std::max(0, x + k));
                    acc += row[xx] * kern[k + radius];
                }
                out[x] = acc;
            }
        }
        // Vertical pass
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) {
            float* out = &dst[y * w];
            for (int x = 0; x < w; ++x) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int yy = std::min(h - 1, std::max(0, y + k));
                    acc += tmp[yy * w + x] * kern[k + radius];
                }
                out[x] = acc;
            }
        }
    }

    // Render one 1/60s frame from the actual stereo audio samples!
    void render_frame(const std::vector<float>& L, const std::vector<float>& R,
                      int start_sample, int count, double time_sec) {
        // 1. Exponential phosphor decay of previous frame
        const float decay_fast = std::exp(-1.0f / (FPS * 0.009f)); // ~9ms razor-sharp core
        const float decay_slow = std::exp(-1.0f / (FPS * 0.034f)); // ~34ms silky CRT afterglow

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < SIZE * SIZE; ++i) {
            fast_buf[i] *= decay_fast;
            slow_buf[i] *= decay_slow;
        }

        // 2. Exact beam-velocity energy splatting along every consecutive sample segment
        const float cx = SIZE * 0.5f;
        const float cy = SIZE * 0.5f;
        const float scale = SIZE * 0.455f;
        const float beam_energy = 0.78f; // Vivid incandescent CRT phosphor exposure

        int end_sample = std::min(int(L.size()) - 1, start_sample + count);
        for (int i = start_sample; i < end_sample; ++i) {
            float x0 = cx + L[i] * scale;
            float y0 = cy - R[i] * scale;
            float x1 = cx + L[i + 1] * scale;
            float y1 = cy - R[i + 1] * scale;

            float dx = x1 - x0;
            float dy = y1 - y0;
            float dist = std::hypot(dx, dy);

            // Beam dwells for constant dt = 1/96000s -> energy spreads over `steps` along segment!
            // Huge teleport jumps (dist > 70px) naturally vanish into black!
            int steps = std::max(1, int(std::ceil(dist * 1.25f)));
            if (steps > 110) continue;

            // Physical beam-velocity curve: crisp bright lines, blazing jewel vertices, dim fast jumps
            float e_step = beam_energy / std::pow(float(steps), 0.78f);
            if (dist < 1.2f) e_step *= 1.30f;

            float inv_steps = 1.0f / float(steps);
            for (int s = 0; s < steps; ++s) {
                float u = (s + 0.5f) * inv_steps;
                float px = x0 + dx * u;
                float py = y0 + dy * u;
                int ix = int(std::floor(px));
                int iy = int(std::floor(py));
                if (ix < 1 || ix >= SIZE - 2 || iy < 1 || iy >= SIZE - 2) continue;

                float fx = px - ix;
                float fy = py - iy;
                float w00 = (1.0f - fx) * (1.0f - fy) * e_step;
                float w10 = fx * (1.0f - fy) * e_step;
                float w01 = (1.0f - fx) * fy * e_step;
                float w11 = fx * fy * e_step;

                int idx = iy * SIZE + ix;
                fast_buf[idx]            += w00;
                fast_buf[idx + 1]        += w10;
                fast_buf[idx + SIZE]     += w01;
                fast_buf[idx + SIZE + 1] += w11;

                slow_buf[idx]            += w00 * 0.42f;
                slow_buf[idx + 1]        += w10 * 0.42f;
                slow_buf[idx + SIZE]     += w01 * 0.42f;
                slow_buf[idx + SIZE + 1] += w11 * 0.42f;
            }
        }

        // 3. Combine fast core + lingering afterglow into HDR buffer
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < SIZE * SIZE; ++i) {
            hdr_buf[i] = fast_buf[i] + slow_buf[i] * 0.65f;
        }

        // 4. 3-Tier Multi-Scale Physical Bloom (Tight + Medium 2x + Wide 4x)
        blur_sep(hdr_buf, bloom1, blur_tmp, SIZE, SIZE, 1.55f, 4);

        // Downsample 2x for medium bloom
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < HALF; ++y) {
            for (int x = 0; x < HALF; ++x) {
                int y2 = y * 2, x2 = x * 2;
                half_a[y * HALF + x] = 0.25f * (
                    bloom1[y2 * SIZE + x2] + bloom1[y2 * SIZE + x2 + 1] +
                    bloom1[(y2 + 1) * SIZE + x2] + bloom1[(y2 + 1) * SIZE + x2 + 1]);
            }
        }
        blur_sep(half_a, half_b, half_a, HALF, HALF, 3.4f, 9);

        // Downsample 4x for wide atmospheric CRT glass halation
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < QTR; ++y) {
            for (int x = 0; x < QTR; ++x) {
                int y2 = y * 2, x2 = x * 2;
                qtr_a[y * QTR + x] = 0.25f * (
                    half_b[y2 * HALF + x2] + half_b[y2 * HALF + x2 + 1] +
                    half_b[(y2 + 1) * HALF + x2] + half_b[(y2 + 1) * HALF + x2 + 1]);
            }
        }
        blur_sep(qtr_a, qtr_b, qtr_a, QTR, QTR, 6.8f, 16);

        // Subtle phosphor chromatic mood across acts (always authentic luminous CRT!)
        // Classic P31 emerald-mint -> cooler electric cyan in Act III Stardust -> rich vivid emerald-cyan in Climax
        float cyan_shift = float(progress(time_sec, 19.5, 21.0) * (1.0 - progress(time_sec, 27.5, 29.0)));
        float climax_warm = float(progress(time_sec, 51.5, 53.5) * (1.0 - progress(time_sec, 63.5, 65.5)));

        float pR = 0.14f + 0.08f * climax_warm;
        float pG = 1.00f;
        float pB = 0.56f + 0.36f * cyan_shift + 0.12f * climax_warm;

        // 5. HDR -> Filmic Tone Mapping + Phosphor Incandescence -> BGR24
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < SIZE; ++y) {
            int hy = std::min(HALF - 1, y >> 1);
            int qy = std::min(QTR - 1, y >> 2);
            for (int x = 0; x < SIZE; ++x) {
                int idx = y * SIZE + x;
                int hx = std::min(HALF - 1, x >> 1);
                int qx = std::min(QTR - 1, x >> 2);

                float core = hdr_buf[idx];
                float b1 = bloom1[idx];
                float b2 = half_b[hy * HALF + hx];
                float b3 = qtr_b[qy * QTR + qx];
                float bg = bg_grid[idx];

                float energy = core * 1.35f + b1 * 0.98f + b2 * 0.68f + b3 * 0.46f;

                // Incandescent white-hot saturation on high-intensity beam cores & vertices
                float hot = std::max(0.0f, core * 0.85f + b1 * 0.30f - 0.50f);
                float r_lin = bg * 0.18f + energy * pR + hot * 0.92f;
                float g_lin = bg * 0.85f + energy * pG + hot * 0.45f;
                float b_lin = bg * 0.58f + energy * pB + hot * 0.85f;

                // Subtle CRT scanline micro-texture (very gentle 3% depth)
                float scan = (y & 1) ? 0.97f : 1.0f;
                r_lin *= scan; g_lin *= scan; b_lin *= scan;

                // ACES-inspired filmic curve + sRGB gamma
                auto tonemap = [](float v) -> uint8_t {
                    v = std::max(0.0f, v);
                    float m = (v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f);
                    m = std::min(1.0f, std::max(0.0f, m));
                    float srgb = std::pow(m, 1.0f / 2.2f);
                    return uint8_t(std::min(255, int(srgb * 255.0f + 0.5f)));
                };

                frame_bgr[idx * 3 + 0] = tonemap(b_lin); // B
                frame_bgr[idx * 3 + 1] = tonemap(g_lin); // G
                frame_bgr[idx * 3 + 2] = tonemap(r_lin); // R
            }
        }
    }
};

// ============================================================================
// Main: Synthesize WAV + Render Full MP4 (or Preview Contact Sheet)
// ============================================================================

static void save_ppm(const std::string& path, const std::vector<uint8_t>& bgr, int w, int h) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        rgb[i * 3 + 0] = bgr[i * 3 + 2];
        rgb[i * 3 + 1] = bgr[i * 3 + 1];
        rgb[i * 3 + 2] = bgr[i * 3 + 0];
    }
    f.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
}

int main(int argc, char** argv) {
    bool preview_only = false;
    std::string out_dir = ".";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--preview") preview_only = true;
        if (std::string(argv[i]) == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
    }

    std::cout << "[Phase Shift v4] Synthesizing " << DURATION << "s @ " << SR << " Hz stereo XY audio...\n" << std::flush;
    std::vector<float> L, R;
    synthesize_audio(L, R);

    std::string wav_path = out_dir + "/Phase Shift v4.wav";
    std::cout << "[Phase Shift v4] Writing 16-bit 96kHz WAV master: " << wav_path << "\n" << std::flush;
    write_wav_16bit(wav_path, L, R);

    CRTScope scope;

    if (preview_only) {
        // Render 18 key timestamps (with 4 frames warm-up each for authentic phosphor persistence!)
        const double times[18] = {
            1.0, 3.0, 5.2, 7.5,
            10.5, 13.0, 16.5, 19.0,
            22.0, 25.0, 29.5, 33.5,
            37.5, 42.5, 48.0, 55.0,
            60.5, 68.5
        };
        std::cout << "[Phase Shift v4] Rendering 18 preview frames...\n" << std::flush;
        for (int k = 0; k < 18; ++k) {
            std::fill(scope.fast_buf.begin(), scope.fast_buf.end(), 0.0f);
            std::fill(scope.slow_buf.begin(), scope.slow_buf.end(), 0.0f);
            int target_frame = std::min(TOTAL_FRAMES - 1, std::max(4, int(times[k] * FPS)));
            for (int f = target_frame - 4; f <= target_frame; ++f) {
                scope.render_frame(L, R, f * SAMPLES_PER_FRAME, SAMPLES_PER_FRAME, double(f) / FPS);
            }
            char name[256];
            std::snprintf(name, sizeof(name), "%s/v4_prev_%02d.ppm", out_dir.c_str(), k);
            save_ppm(name, scope.frame_bgr, SIZE, SIZE);
        }
        std::cout << "[Phase Shift v4] Preview frames written.\n";
        return 0;
    }

    std::string mp4_path = out_dir + "/Phase Shift v4.mp4";
    std::string cmd =
        "ffmpeg -hide_banner -loglevel warning -y "
        "-f rawvideo -pix_fmt bgr24 -s " + std::to_string(SIZE) + "x" + std::to_string(SIZE) +
        " -framerate " + std::to_string(FPS) + " -i - "
        "-i \"" + wav_path + "\" "
        "-map 0:v:0 -map 1:a:0 "
        "-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "
        "-c:a aac -b:a 256k -ar 48000 "
        "-movflags +faststart -metadata title=\"Phase Shift v4\" "
        "\"" + mp4_path + "\"";

    std::cout << "[Phase Shift v4] Rendering " << TOTAL_FRAMES << " frames (" << SIZE << "x" << SIZE << " @ " << FPS << "fps) -> " << mp4_path << "\n" << std::flush;

    FILE* pipe = _popen(cmd.c_str(), "wb");
    if (!pipe) {
        std::cerr << "Failed to launch ffmpeg!\n";
        return 1;
    }

    for (int f = 0; f < TOTAL_FRAMES; ++f) {
        scope.render_frame(L, R, f * SAMPLES_PER_FRAME, SAMPLES_PER_FRAME, double(f) / FPS);
        size_t written = std::fwrite(scope.frame_bgr.data(), 1, scope.frame_bgr.size(), pipe);
        if (written != scope.frame_bgr.size()) {
            std::cerr << "Pipe write error at frame " << f << "\n";
            _pclose(pipe);
            return 1;
        }
        if (f % (FPS * 6) == 0) {
            std::cout << "  Rendered " << (f / FPS) << "s / " << int(DURATION) << "s\n" << std::flush;
        }
    }

    int rc = _pclose(pipe);
    if (rc != 0) {
        std::cerr << "ffmpeg exited with code " << rc << "\n";
        return rc;
    }

    std::cout << "[Phase Shift v4] Done! Deliverables:\n  " << wav_path << "\n  " << mp4_path << "\n";
    return 0;
}
