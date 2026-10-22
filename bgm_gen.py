# -*- coding: utf-8 -*-
"""程序化中国风纪录片配乐合成（参考实现）
来源: 《国宝的长征》项目实战封装
方法论: jnMetaCode/ai-shortfilm-prompts (5段式 + 节拍锁定 + 留白结尾)

用法:
1. 修改顶部 SR/DUR/SLOT 适配你的视频
2. 按三幕结构调整编曲部分
3. python bgm_gen.py → bgm_raw.wav
4. ffmpeg 归一化: ffmpeg -i bgm_raw.wav -af loudnorm=I=-18:TP=-2:LRA=11 -c:a pcm_s24le bgm.wav
"""
import numpy as np
from scipy import signal as sig
import soundfile as sf

SR = 48000
DUR = 177.245          # ← 改成你的视频总时长（秒）
N = int(DUR * SR)
SLOT = DUR / 22.0      # ← 改成 总时长/镜数
BEAT = SLOT / 8.0      # 节拍锁定: 每镜=2小节4/4 → 8拍
BAR = 4 * BEAT
rng = np.random.default_rng(42)

mixL = np.zeros(N + SR * 4)
mixR = np.zeros(N + SR * 4)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def place(sig_l, sig_r, t0, amp=1.0):
    i0 = int(t0 * SR)
    n = len(sig_l)
    if i0 < 0 or i0 + n > len(mixL):
        n = min(n, len(mixL) - i0)
        if n <= 0:
            return
        sig_l, sig_r = sig_l[:n], sig_r[:n]
    mixL[i0:i0 + n] += sig_l * amp
    mixR[i0:i0 + n] += sig_r * amp


def env_asr(n, a, r, sustain=1.0):
    e = np.full(n, sustain, dtype=np.float64)
    na, nr = int(a * SR), int(r * SR)
    na, nr = min(na, n), min(nr, n)
    if na > 0:
        e[:na] *= np.sin(np.linspace(0, np.pi / 2, na)) ** 2
    if nr > 0:
        e[-nr:] *= np.cos(np.linspace(0, np.pi / 2, nr)) ** 2
    return e


def stereo(x, width=0.12):
    d = int(width * SR)
    r = np.zeros_like(x)
    if d > 0:
        r[d:] = x[:-d]
    else:
        r[:] = x
    return x, 0.92 * r


# ---------- 乐器合成 ----------
def guqin(m, dur=4.0):
    """Karplus-Strong 拨弦: 古琴散音质感"""
    f = hz(m)
    p = max(2, int(SR / f))
    burst = sig.lfilter(*sig.butter(2, min(1800, f * 4) / (SR / 2), 'low'),
                        rng.standard_normal(p))
    n = int(dur * SR)
    out = np.zeros(n + p)
    buf = burst.copy()
    j = 0
    for i in range(n):
        out[i] = buf[j]
        buf[j] = 0.9965 * 0.5 * (buf[j] + buf[(j + 1) % p])
        j = (j + 1) % p
    out = out[:n]
    out[:60] *= np.linspace(0, 1, 60)
    out *= np.exp(-np.arange(n) / (SR * 1.4)) * 0.9
    return stereo(out)


def bell(m, dur=5.0):
    """加法合成编钟: 非谐分音+分级衰减"""
    f = hz(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    partials = [(1.0, 1.0, 5.0), (2.0, 0.45, 2.6), (2.98, 0.28, 1.4), (4.21, 0.14, 0.7)]
    out = np.zeros(n)
    for ratio, amp, dec in partials:
        out += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / dec)
    out[:80] *= np.linspace(0, 1, 80)
    return stereo(out, 0.2)


def taiko(amp_curve=90.0, dur=0.9, skin=0.5):
    """中国大鼓: 正弦扫频鼓身+噪声鼓皮"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = amp_curve * np.exp(-t / 0.06) + 42
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    ns = rng.standard_normal(n) * np.exp(-t / 0.02)
    ns = sig.lfilter(*sig.butter(2, 900 / (SR / 2), 'low'), ns) * skin
    return stereo((body + ns) * 0.9)


def pad_note(ms, dur, a=1.8, r=1.6):
    """弦乐垫: 多分音×3组微失谐, 慢起慢收"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for m in ms:
        f0 = hz(m)
        for det, w in [(-7, 0.4), (0.0, 0.5), (7, 0.4)]:
            f = f0 * 2 ** (det / 1200)
            ph = 2 * np.pi * f * t + rng.uniform(0, 6.28)
            lfo = 1 + 0.04 * np.sin(2 * np.pi * rng.uniform(0.1, 0.25) * t + rng.uniform(0, 6.28))
            for k, ak in [(1, 1.0), (2, 0.32), (3, 0.12), (4, 0.05)]:
                out += w * ak * np.sin(k * ph) * lfo / len(ms)
    out /= np.max(np.abs(out) + 1e-9)
    return stereo(out * env_asr(n, a, r))


def strings_pluck(m, dur=0.55):
    """低音弦乐短音(大提琴固定音型)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    out = np.zeros(n)
    for det in (-5, 0, 5):
        ff = f * 2 ** (det / 1200)
        for k, ak in [(1, 1.0), (2, 0.45), (3, 0.18), (4, 0.08)]:
            out += ak * np.sin(2 * np.pi * ff * k * t + rng.uniform(0, 6.28)) / 3
    out *= env_asr(n, 0.035, 0.30) * np.exp(-t / 0.35)
    return stereo(out, 0.07)


def xiao(m, dur=4.0, vib=5.2):
    """箫: 正弦+颤音+气声"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m) * (1 + 0.004 * np.sin(2 * np.pi * vib * t) * env_asr(n, 1.0, 1.0, 1.0))
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.sin(ph) + 0.30 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    breath = sig.lfilter(*sig.butter(2, [1800 / (SR / 2), 5200 / (SR / 2)], 'band'),
                         rng.standard_normal(n)) * 0.05
    return stereo((out + breath) * env_asr(n, 0.55, 0.8) / 1.4)


def drone(ms, dur, a=4.0, r=3.0):
    return pad_note(ms, dur, a=a, r=r)


# ---------- 混响 ----------
def build_ir(seconds=1.7, predelay=0.024):
    n = int(seconds * SR) + int(predelay * SR)
    t = np.arange(n) / SR
    irs = []
    for _ in range(2):
        w = rng.standard_normal(n) * np.exp(-np.maximum(t - predelay, 0) / 0.42)
        w = sig.lfilter(*sig.butter(2, 5500 / (SR / 2), 'low'), w)
        w[: int(predelay * SR)] = 0
        irs.append(w / (np.linalg.norm(w) + 1e-9))
    return irs


IR_L, IR_R = build_ir()

# ---------- 编曲(节拍锁定时间轴) ----------
# T(b) = 第b拍的秒数。改编曲时用拍号而非秒，保证节拍锁定。
T = lambda b: b * BEAT + rng.uniform(-0.012, 0.012)

# ===== 幕一·序(压抑) =====
place(*drone([38, 33], 41.5, a=5.0, r=2.0), 0.0, 0.085)
place(*pad_note([33, 38], 40.5, a=6.0, r=2.5), 0.8, 0.12)
wind_n = sig.lfilter(*sig.butter(2, 350 / (SR / 2), 'low'), rng.standard_normal(N))
wind_lfo = 0.55 + 0.45 * np.sin(2 * np.pi * 0.09 * np.arange(N) / SR + 1.2)
mixL[:N] += (wind_n * wind_lfo * 0.030)
mixR[:N] += (np.roll(wind_n, 1700) * np.roll(wind_lfo, 1700) * 0.024)

for b in [2, 3, 4, 5, 6, 7, 8, 9]:
    place(*taiko(62, 0.8, 0.25), T(b * 4), 0.18)          # 心跳
place(*taiko(85, 1.4, 0.7), T(6), 0.30)
place(*taiko(85, 1.4, 0.7), T(30), 0.28)
for b, m in [(10, 50), (12.5, 53), (14, 55), (16.5, 53), (18, 50), (24, 48), (26, 50), (28.5, 45)]:
    place(*guqin(m), T(b), 0.85)
place(*bell(62), T(24), 0.55)
place(*pad_note([38, 41, 45], 9.0, a=4.5, r=2.5), T(31), 0.65)
place(*taiko(95, 1.2, 0.6), T(37.5), 0.5)
place(*taiko(95, 1.2, 0.6), T(39), 0.62)
place(*taiko(100, 1.6, 0.9), T(40), 1.0)                  # 转场重击

# ===== 幕二·征途(行进) =====
full = {10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 30, 31, 32, 33}
thin = {18, 19, 24, 25, 26, 27, 28, 29}
for b in range(10, 34):
    t0 = b * 4
    if b in full:
        pat = [(0, 1.00), (1.5, 0.50), (2, 0.72), (3.5, 0.45)]
        if b >= 30:
            pat += [(1, 0.42), (3, 0.52)]
    else:
        pat = [(0, 0.55)]
    for off, a in pat:
        place(*taiko(88, 0.7, 0.5), T(t0 + off), a)
    if b in full:
        for off, m in [(0, 38), (1, 38), (2, 33), (3, 36)]:
            place(*strings_pluck(m), T(t0 + off), 0.78)
    else:
        for off, m in [(0, 38), (2, 36)]:
            place(*strings_pluck(m), T(t0 + off), 0.48)

place(*drone([38, 33], 97.5, a=2.5, r=2.0), T(40), 0.075)
place(*pad_note([38, 41, 45], 32.2, a=3.0, r=2.0), T(40), 0.42)
place(*pad_note([46, 50, 53], 8.6, a=2.5, r=2.0), T(128), 0.62)
# 推峰滚奏
roll_n = int(8.5 * SR)
roll_env = np.sin(np.linspace(0, np.pi / 2, roll_n)) ** 1.5
roll = sig.lfilter(*sig.butter(2, 180 / (SR / 2), 'low'), rng.standard_normal(roll_n)) * roll_env * 0.22
place(*stereo(roll, 0.15), T(124), 1.0)
# 古琴旋律
for b, m in [(48, 62), (54, 65), (56, 67), (64, 69), (70, 67), (72, 65),
             (80, 62), (86, 65), (88, 67), (94, 69), (96, 72), (102, 69)]:
    place(*guqin(m), T(b), 0.8)
place(*bell(69), T(74), 0.5)
place(*xiao(65, 4.2), T(98), 0.6)
place(*xiao(67, 5.2), T(104), 0.62)
place(*bell(69), T(112), 0.42)
place(*xiao(69, 7.0, vib=4.6), T(112.5), 0.6)
place(*bell(74), T(116), 0.4)
for i in range(16):
    place(*guqin(62, dur=1.2), T(120 + i * 0.5), 0.45 + 0.12 * (i % 2))
for b, m in [(128, 62), (132, 65), (134, 67), (135.5, 69)]:
    place(*guqin(m), T(b), 0.85)

# ===== 幕三·归来(解决+留白) =====
place(*taiko(96, 1.8, 0.8), T(136), 1.12)
place(*drone([38, 45, 50], 41.5, a=2.0, r=4.0), T(136), 0.09)
place(*pad_note([38, 45, 50, 54], 24.0, a=2.5, r=4.0), T(136), 0.62)
place(*bell(74), T(136.5), 0.7)
for b in range(34, 38):
    place(*taiko(80, 0.7, 0.4), T(b * 4), 0.4)
for b, m in [(144, 62), (148, 64), (150, 66), (152, 69), (156, 71), (157.5, 69)]:
    place(*guqin(m), T(b), 0.78)
for b, m in [(146.5, 66), (147.5, 69), (151.5, 74)]:
    place(*guqin(m, dur=2.0), T(b), 0.55)
place(*xiao(66, 6.0), T(158), 0.55)
place(*bell(74), T(160), 0.42)
place(*bell(69), T(166), 0.34)
place(*pad_note([50, 57], 16.0, a=3.0, r=5.0), T(160), 0.48)
place(*guqin(62, dur=5.0), T(170), 0.7)
place(*drone([38], 7.4, a=2.0, r=5.0), T(170), 0.06)

# ---------- 总线 ----------
mix = np.stack([mixL[:N], mixR[:N]], axis=1)
wet_l = sig.fftconvolve(mix[:, 0], IR_L)[:N]
wet_r = sig.fftconvolve(mix[:, 1], IR_R)[:N]
mix = mix[:N] * 0.82 + np.stack([wet_l, wet_r], axis=1) * 0.30

fade = np.ones(N)
nf = int(3.5 * SR)
fade[-nf:] = np.cos(np.linspace(0, np.pi / 2, nf)) ** 2
fade[: int(0.3 * SR)] *= np.linspace(0, 1, int(0.3 * SR))
mix *= fade[:, None]

peak = np.max(np.abs(mix))
mix = mix / peak * 0.84
sf.write('bgm_raw.wav', mix.astype(np.float32), SR)
print(f'BGM_DONE {N/SR:.3f}s peak={20*np.log10(peak):.2f}dBFS', flush=True)
