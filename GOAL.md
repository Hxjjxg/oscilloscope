# Phase Shift v3 项目说明

《Phase Shift v3》是一部约 60 秒的原创 XY 视听作品。立体声母带的左、右声道分别对应 X、Y。作品探索音乐的层次、几何形体的连续动作，以及两者在乐句转折处的关系。项目保留了 v1 和 v2 的成品与生成器。

本文供了解项目和回看历史版本。每轮创作以当次委托为准，形式、工具和试验方法可随问题调整。有一份针对 `559a08b` 的单次创作指示，仅在该轮任务明确采用时生效。[Phase Shift v3 单次创作指示](<Phase Shift v3 单次创作指示.md>)

## 版本记录

| Git 提交 | 用途 |
| --- | --- |
| `559a08b` | 本轮归档时的成片。恢复较明亮的主题音色，让伴奏自然延音，并在约 25 秒引入慢起音的笛声。已核对音画同步，实际舒适度和音色转折仍待试听。 |
| `ca01ddd` | 较早的双立方体点阵版。用来比较较高的音区和约 25 秒的音色变化。 |
| `cf8d1b8` | 用户贴出的临时目录 MP4 所对应的版本。用来比较双立方体的动作稳定性，以及用户指出的刺耳感和较弱的音色转折。 |
| `8e87282` | 帆船版。可回看曾出现轮廓叠影的问题。 |

实现细节和测量结果见制作记录。[音色修订与复核](<Phase Shift v3 final review.md>)

## 参考资料

- Atom Delta 的《Oscillofun》在 0:17.4–0:20.7 展示了内框、外框与连接棱如何在旋转和换音时保持可辨。1:16.2–1:19.6 有平面缩放与错位重复。[本地音频](<references/oscillofun-wave.flac>)
- BUS ERROR Collective 的《Primer》在约 0:39.8–0:44 由窄带打开成球壳，约 0:48–0:53 出现向留白处移出的第二个形体，2:41.5–2:48.1 有外壳围绕同一中心展开的动作。[本地音频](<references/Primer Final.flac>)

[XY 动作研究](<XY 动作研究.md>) · [参考抽帧](<references/XY motion study.png>) · [网络研究](<网络研究.md>)

## 成品与源码

v3 的母带为 48 kHz、24 位立体声 PCM，视频为 1080 × 1080、60 fps。生成器依赖 Python 3.10+、NumPy、SciPy、SoundFile、OpenCV 和带有 libx264、AAC 编码器的 FFmpeg。在项目目录运行 `python make_phase_shift_v3.py` 可重建两份成品。

[WAV 母带](<Phase Shift v3.wav>) · [MP4 成片](<Phase Shift v3.mp4>) · [独立生成器](make_phase_shift_v3.py)
