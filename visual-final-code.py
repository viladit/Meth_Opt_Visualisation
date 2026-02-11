from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


# ============================================================
# 0) Цвета / стиль
# ============================================================
ITMO_BLUE   = "#1946BA"
ITMO_CYAN   = "#3DB5E6"
ITMO_PINK   = "#EC0B43"
ITMO_ORANGE = "#FF8400"
ITMO_LIME   = "#94D600"

DARK = "#111827"
GRID = "#E5E7EB"
BG   = "#FFFFFF"


# ============================================================
# 1) FFmpeg
# ============================================================
FFMPEG_PATH: Optional[str] = r"/Users/viladit/PycharmProjects/MethOpt/ffmpeg"

def ensure_ffmpeg():
    if FFMPEG_PATH:
        p = Path(FFMPEG_PATH)
        if not p.exists():
            raise RuntimeError(f"FFMPEG_PATH указан, но файла нет: {FFMPEG_PATH}")
        mpl.rcParams["animation.ffmpeg_path"] = str(p)
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден: укажи FFMPEG_PATH или добавь ffmpeg в PATH.")

def ffmpeg_bin() -> str:
    p = mpl.rcParams.get("animation.ffmpeg_path")
    if p:
        return str(p)
    return "ffmpeg"


# ============================================================
# 2) Глобальные настройки
# ============================================================
OUT_DIR = Path("videos_itmo_v6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 30
DPI = 140

A1D = 0.0
B1D = 4.0
EPS_1D = 1e-3

X0_NEWTON = 0.8

X1_QUAD = 2.0
H_QUAD  = 0.6

X0_2D = np.array([3.0, -2.0], dtype=float)
EPS_2D = 1e-3
MAX_IT_2D = 200


# ============================================================
# 3) Утилиты
# ============================================================
def ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)

def norm2(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v*v)))

def bracket_minimum(phi: Callable[[float], float], step: float = 1.0, grow: float = 2.0,
                    max_expand: int = 60) -> Tuple[float, float, float]:
    x0 = 0.0
    f0 = phi(x0)

    x1 = x0 + step
    f1 = phi(x1)

    if f1 > f0:
        step = -step
        x1 = x0 + step
        f1 = phi(x1)
        if f1 > f0:
            a, b = (x1, x0) if x1 < x0 else (x0, x1)
            return a, b, x0

    for _ in range(max_expand):
        step *= grow
        x2 = x1 + step
        f2 = phi(x2)
        if f2 > f1:
            a, b = (x0, x2) if x0 < x2 else (x2, x0)
            return a, b, x1
        x0, f0, x1, f1 = x1, f1, x2, f2

    a, b = (x0, x1) if x0 < x1 else (x1, x0)
    return a, b, x1

def golden_section_min(phi: Callable[[float], float], a: float, b: float, eps: float = 1e-6, max_iter: int = 200) -> float:
    gr = (math.sqrt(5) - 1) / 2
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1 = phi(x1)
    f2 = phi(x2)
    for _ in range(max_iter):
        if (b - a) <= eps:
            break
        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = a + gr * (b - a)
            f2 = phi(x2)
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = b - gr * (b - a)
            f1 = phi(x1)
    return (a + b) / 2

def adaptive_frames(n_steps: int, fps: int, frames_per_step: int, hold_frames: int, max_seconds: int = 35) -> Tuple[int, int]:
    if n_steps <= 0:
        return frames_per_step, hold_frames
    max_total_frames = fps * max_seconds
    base = n_steps * (frames_per_step + hold_frames)
    if base <= max_total_frames:
        return frames_per_step, hold_frames
    scale = max_total_frames / base
    fpss = max(3, int(frames_per_step * scale))
    hfs = max(1, int(hold_frames * scale))
    return fpss, hfs

def progress(tag: str, i: int, n: int, each: int = 8):
    if n <= 0:
        return
    if n <= 25:
        pct = int(round(100 * (i+1) / n))
        print(f"[{tag}] {i+1}/{n} ({pct}%)")
        return
    if i == 0 or i == n-1 or (i % max(1, n//each) == 0):
        pct = int(round(100 * (i+1) / n))
        print(f"[{tag}] {i+1}/{n} ({pct}%)")


# ============================================================
# 4) 1D функции
# ============================================================
def f1d_base(x: float) -> float:
    return (x - 1.7)**2 + 0.2*math.sin(3*x) + 1.0

def df1d_base(x: float) -> float:
    return 2*(x - 1.7) + 0.6*math.cos(3*x)

def d2f1d_base(x: float) -> float:
    return 2 - 1.8*math.sin(3*x)

def f1d_chords(x: float) -> float:
    return (x - 2.0)**4 + 0.15*(x - 2.0)**2 + 0.10*math.sin(3.0*x) + 0.12*x

def df1d_chords(x: float) -> float:
    return 4*(x - 2.0)**3 + 0.30*(x - 2.0) + 0.30*math.cos(3.0*x) + 0.12

def d2f1d_chords(x: float) -> float:
    return 12*(x - 2.0)**2 + 0.30 - 0.90*math.sin(3.0*x)

def f1d_quad(x: float) -> float:
    return (x - 2.3)**4 + 0.2*(x - 2.3)**2 + 0.15*x + 0.08*math.sin(4.0*x)


# ============================================================
# 5) 2D функция (уровни)
# ============================================================
def f2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x - 1.2)**2 + 2.0*(y + 0.7)**2 + 0.7*np.sin(1.5*x) + 0.4*np.cos(1.2*y)

def grad2d(x: float, y: float) -> np.ndarray:
    gx = 2*(x - 1.2) + 0.7*1.5*np.cos(1.5*x)
    gy = 4*(y + 0.7) - 0.4*1.2*np.sin(1.2*y)
    return np.array([gx, gy], dtype=float)


# ============================================================
# 6) Шаги методов
# ============================================================
@dataclass
class HalvingStep:
    a: float; b: float
    x1: float; x2: float
    y1: float; y2: float
    update: str

@dataclass
class GoldenStep:
    a: float; b: float
    x1: float; x2: float
    y1: float; y2: float
    reuse: str

@dataclass
class ChordStep:
    a: float; b: float
    ga: float; gb: float
    x: float; gx: float

@dataclass
class NewtonStep:
    x: float
    gx: float
    gpx: float
    x_next: float

@dataclass
class QuadStep:
    x1: float; x2: float; x3: float
    f1: float; f2: float; f3: float
    xbar: float; fbar: float
    xmin: float; fmin: float


# ============================================================
# 7) Генерация шагов
# ============================================================
def halving_steps(a0: float, b0: float, eps: float, max_iter: int = 100000) -> List[HalvingStep]:
    a, b = a0, b0
    steps: List[HalvingStep] = []
    for _ in range(max_iter):
        x1 = (a + b - eps) / 2
        x2 = (a + b + eps) / 2
        y1 = f1d_base(x1)
        y2 = f1d_base(x2)
        if y1 > y2:
            update = "a=x1"
            a = x1
        else:
            update = "b=x2"
            b = x2
        steps.append(HalvingStep(a=a, b=b, x1=x1, x2=x2, y1=y1, y2=y2, update=update))
        if (b - a) <= 2*eps:
            break
    return steps

def golden_steps(a0: float, b0: float, eps: float, max_iter: int = 100000) -> List[GoldenStep]:
    gr = (math.sqrt(5) - 1) / 2
    a, b = a0, b0
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    y1 = f1d_base(x1)
    y2 = f1d_base(x2)

    steps: List[GoldenStep] = []
    for _ in range(max_iter):
        if y1 > y2:
            reuse = "x2 → x1"
            a = x1
            x1, y1 = x2, y2
            x2 = a + gr * (b - a)
            y2 = f1d_base(x2)
        else:
            reuse = "x1 → x2"
            b = x2
            x2, y2 = x1, y1
            x1 = b - gr * (b - a)
            y1 = f1d_base(x1)

        steps.append(GoldenStep(a=a, b=b, x1=x1, x2=x2, y1=y1, y2=y2, reuse=reuse))
        if (b - a) <= eps:
            break
    return steps

def chords_on_derivative(a0: float, b0: float, eps: float, max_iter: int = 100000) -> List[ChordStep]:
    g = df1d_chords
    a, b = a0, b0
    ga = g(a); gb = g(b)
    if ga * gb > 0:
        raise ValueError("Для метода хорд нужно g(a)*g(b)<0. Подбери [a,b].")
    steps: List[ChordStep] = []
    for _ in range(max_iter):
        x = (a * gb - b * ga) / (gb - ga)
        gx = g(x)
        steps.append(ChordStep(a=a, b=b, ga=ga, gb=gb, x=x, gx=gx))
        if abs(gx) <= eps or abs(b - a) <= eps:
            break
        if ga * gx < 0:
            b, gb = x, gx
        else:
            a, ga = x, gx
    return steps

def newton_on_derivative(x0: float, eps: float, min_steps: int = 6, max_iter: int = 100000) -> List[NewtonStep]:
    g = df1d_chords
    gp = d2f1d_chords
    x = x0
    steps: List[NewtonStep] = []
    for _ in range(max_iter):
        gx = g(x)
        gpx = gp(x)
        if abs(gpx) < 1e-14:
            break
        x_next = x - gx / gpx
        steps.append(NewtonStep(x=x, gx=gx, gpx=gpx, x_next=x_next))
        if len(steps) >= min_steps and abs(x_next - x) <= eps:
            break
        x = x_next
    return steps

def quad_steps(a: float, b: float, x_center: float, h: float, eps: float,
              min_steps: int = 10, max_iter: int = 5000) -> List[QuadStep]:
    def clamp(x: float) -> float:
        return max(a, min(b, x))

    x2 = clamp(x_center)
    hh = abs(h) if abs(h) > 1e-12 else 0.2
    x1 = clamp(x2 - hh)
    x3 = clamp(x2 + hh)
    x1, x2, x3 = sorted([x1, x2, x3])

    steps: List[QuadStep] = []
    prev_best: Optional[float] = None

    for _ in range(max_iter):
        f1 = f1d_quad(x1); f2 = f1d_quad(x2); f3 = f1d_quad(x3)

        denom = (x2 - x3)*f1 + (x3 - x1)*f2 + (x1 - x2)*f3
        numer = (x2**2 - x3**2)*f1 + (x3**2 - x1**2)*f2 + (x1**2 - x2**2)*f3
        if abs(denom) < 1e-14:
            xbar = x2
        else:
            xbar = 0.5 * numer / denom
        xbar = clamp(xbar)
        fbar = f1d_quad(xbar)

        candidates = [(x1, f1), (x2, f2), (x3, f3), (xbar, fbar)]
        xmin, fmin = min(candidates, key=lambda t: t[1])

        steps.append(QuadStep(x1=x1, x2=x2, x3=x3,
                              f1=f1, f2=f2, f3=f3,
                              xbar=xbar, fbar=fbar,
                              xmin=xmin, fmin=fmin))

        if len(steps) >= min_steps:
            if prev_best is not None and abs(xmin - prev_best) <= eps:
                break
            if abs(x3 - x1) <= eps:
                break
        prev_best = xmin

        pts = sorted({round(x, 14) for x, _ in candidates})
        idx = min(range(len(pts)), key=lambda i: abs(pts[i] - xmin))
        left = pts[max(0, idx-1)]
        mid = pts[idx]
        right = pts[min(len(pts)-1, idx+1)]
        if not (left < mid < right):
            left = clamp(xmin - hh)
            right = clamp(xmin + hh)
            mid = xmin
            left, mid, right = sorted([left, mid, right])

        if abs(right - left) < 1e-6:
            hh *= 0.5
            left = clamp(xmin - hh)
            right = clamp(xmin + hh)
            mid = xmin
            left, mid, right = sorted([left, mid, right])

        x1, x2, x3 = left, mid, right

    return steps


# ============================================================
# 8) 2D методы (траектории)
# ============================================================
@dataclass
class Path2D:
    points: List[np.ndarray]
    grads: List[np.ndarray]
    step_info: List[str]

def coord_descent_2d(x0: np.ndarray, eps: float, max_iter: int = 200) -> Path2D:
    x = x0.astype(float).copy()
    pts = [x.copy()]
    grads = [grad2d(x[0], x[1])]
    info = ["start"]
    for k in range(max_iter):
        x_prev = x.copy()
        for i in range(2):
            def phi(t: float) -> float:
                xx = x.copy()
                xx[i] = t
                return float(f2d(xx[0], xx[1]))
            a, b, _ = bracket_minimum(phi, step=1.0)
            tmin = golden_section_min(phi, a, b, eps=1e-5, max_iter=120)
            x[i] = tmin
        pts.append(x.copy())
        grads.append(grad2d(x[0], x[1]))
        info.append("coord update")
        if norm2(x - x_prev) < eps:
            break
    return Path2D(points=pts, grads=grads, step_info=info)

def gradient_descent_2d(x0: np.ndarray, eps: float, alpha: float = 0.10, max_iter: int = 200) -> Path2D:
    x = x0.astype(float).copy()
    pts = [x.copy()]
    grads = [grad2d(x[0], x[1])]
    info = ["start"]
    for k in range(max_iter):
        g = grad2d(x[0], x[1])
        x_next = x - alpha * g
        pts.append(x_next.copy())
        grads.append(grad2d(x_next[0], x_next[1]))
        info.append(f"alpha={alpha:.3f}")
        if norm2(x_next - x) < eps:
            break
        x = x_next
    return Path2D(points=pts, grads=grads, step_info=info)

def steepest_descent_2d(x0: np.ndarray, eps: float, max_iter: int = 200) -> Path2D:
    x = x0.astype(float).copy()
    pts = [x.copy()]
    grads = [grad2d(x[0], x[1])]
    info = ["start"]
    for k in range(max_iter):
        g = grad2d(x[0], x[1])
        d = -g
        def phi(lam: float) -> float:
            xx = x + lam*d
            return float(f2d(xx[0], xx[1]))
        a, b, _ = bracket_minimum(phi, step=0.3)
        lam = golden_section_min(phi, a, b, eps=1e-5, max_iter=160)
        x_next = x + lam*d
        pts.append(x_next.copy())
        grads.append(grad2d(x_next[0], x_next[1]))
        info.append(f"lambda*={lam:.4f}")
        if norm2(x_next - x) < eps:
            break
        x = x_next
    return Path2D(points=pts, grads=grads, step_info=info)


# ============================================================
# 9) Отрисовка 1D — общие
# ============================================================
def setup_ax_1d(ax: plt.Axes, title: str, xlabel: str = "x", ylabel: str = "f(x)"):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=DARK, fontsize=12, fontweight="bold")

def plot_func(ax: plt.Axes, func: Callable[[float], float], xlim: Tuple[float, float], n: int = 800):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.array([func(float(x)) for x in xs])
    (line,) = ax.plot(xs, ys, color=ITMO_BLUE, linewidth=2.4, alpha=0.95)
    return line

def update_func_line(line, func: Callable[[float], float], xlim: Tuple[float, float], n: int = 800):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.array([func(float(x)) for x in xs])
    line.set_data(xs, ys)

def zoom_xlim(ax: plt.Axes, left: float, right: float, pad_frac: float = 0.35, min_width: float = 0.02):
    mn = min(left, right)
    mx = max(left, right)
    w = max(min_width, mx - mn)
    pad = pad_frac * w
    ax.set_xlim(mn - pad, mx + pad)

def zoom_ylim_from_xlim(ax: plt.Axes, func: Callable[[float], float], pad_frac: float = 0.20):
    xl = ax.get_xlim()
    xs = np.linspace(xl[0], xl[1], 900)
    ys = np.array([func(float(x)) for x in xs])
    ymin = float(np.min(ys)); ymax = float(np.max(ys))
    h = max(1e-9, ymax - ymin)
    ax.set_ylim(ymin - pad_frac*h, ymax + pad_frac*h)


# ============================================================
# 10) Видео: половинное деление
# ============================================================
def render_halving(steps: List[HalvingStep], out_file: Path):
    if not steps:
        return
    fpss, hfs = adaptive_frames(len(steps), FPS, frames_per_step=10, hold_frames=6)

    a0 = steps[0].a
    b0 = steps[0].b
    xlim0 = (a0 - 0.4*(b0-a0), b0 + 0.4*(b0-a0))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    setup_ax_1d(ax, "1) Метод половинного деления")
    f_line = plot_func(ax, f1d_base, xlim0)

    v_a = ax.axvline(steps[0].a, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)
    v_b = ax.axvline(steps[0].b, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)
    p1, = ax.plot([], [], "o", color=ITMO_PINK, markersize=6)
    p2, = ax.plot([], [], "o", color=ITMO_PINK, markersize=6)

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=FPS, bitrate=1800,
        extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    )

    with writer.saving(fig, str(out_file), dpi=DPI):
        for i, st in enumerate(steps):
            progress("halving", i, len(steps))
            zoom_xlim(ax, st.a, st.b, pad_frac=0.35, min_width=0.02)
            zoom_ylim_from_xlim(ax, f1d_base)
            update_func_line(f_line, f1d_base, ax.get_xlim())

            prev = steps[i-1] if i > 0 else st

            for j in range(fpss):
                tt = ease(j/(fpss-1) if fpss > 1 else 1.0)
                a_vis = prev.a + (st.a - prev.a)*tt
                b_vis = prev.b + (st.b - prev.b)*tt
                v_a.set_xdata([a_vis, a_vis])
                v_b.set_xdata([b_vis, b_vis])

                p1.set_data([st.x1], [st.y1])
                p2.set_data([st.x2], [st.y2])
                txt.set_text(
                    f"k={i+1}/{len(steps)}\n[a,b]=[{st.a:.5f},{st.b:.5f}]\n"
                    f"x1={st.x1:.5f}, x2={st.x2:.5f}\n"
                    f"f(x1)={st.y1:.5f}, f(x2)={st.y2:.5f}\n"
                    f"обновление: {st.update}"
                )
                writer.grab_frame()

            for _ in range(hfs):
                writer.grab_frame()

    plt.close(fig)


# ============================================================
# 11) Видео: золотое сечение (видно перенос точки)
# ============================================================
def render_golden(steps: List[GoldenStep], out_file: Path):
    if not steps:
        return
    fpss, hfs = adaptive_frames(len(steps), FPS, frames_per_step=12, hold_frames=7)

    a0 = steps[0].a
    b0 = steps[0].b
    xlim0 = (a0 - 0.4*(b0-a0), b0 + 0.4*(b0-a0))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    setup_ax_1d(ax, "2) Метод золотого сечения (виден перенос точки)")
    f_line = plot_func(ax, f1d_base, xlim0)

    v_a = ax.axvline(steps[0].a, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)
    v_b = ax.axvline(steps[0].b, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)

    # 2 точки разного цвета (роль x1 и x2)
    p1, = ax.plot([], [], "o", color=ITMO_PINK, markersize=7)    # x1
    p2, = ax.plot([], [], "o", color=ITMO_ORANGE, markersize=7)  # x2
    t1 = ax.text(0, 0, "x1", color=DARK, fontsize=10, ha="left", va="bottom")
    t2 = ax.text(0, 0, "x2", color=DARK, fontsize=10, ha="left", va="bottom")

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=FPS, bitrate=1800,
        extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    )

    with writer.saving(fig, str(out_file), dpi=DPI):
        for i, st in enumerate(steps):
            progress("golden", i, len(steps))
            zoom_xlim(ax, st.a, st.b, pad_frac=0.35, min_width=0.02)
            zoom_ylim_from_xlim(ax, f1d_base)
            update_func_line(f_line, f1d_base, ax.get_xlim())

            prev = steps[i-1] if i > 0 else st
            for j in range(fpss):
                t = j/(fpss-1) if fpss > 1 else 1.0
                tt = ease(t)

                a_vis = prev.a + (st.a - prev.a)*tt
                b_vis = prev.b + (st.b - prev.b)*tt
                v_a.set_xdata([a_vis, a_vis])
                v_b.set_xdata([b_vis, b_vis])

                x1_vis = prev.x1 + (st.x1 - prev.x1)*tt
                x2_vis = prev.x2 + (st.x2 - prev.x2)*tt
                y1_vis = f1d_base(x1_vis)
                y2_vis = f1d_base(x2_vis)

                if st.reuse == "x2 → x1":
                    p1.set_alpha(0.6 + 0.4*math.sin(math.pi*t))
                    p2.set_alpha(0.95)
                else:
                    p2.set_alpha(0.6 + 0.4*math.sin(math.pi*t))
                    p1.set_alpha(0.95)

                p1.set_data([x1_vis], [y1_vis])
                p2.set_data([x2_vis], [y2_vis])
                t1.set_position((x1_vis, y1_vis))
                t2.set_position((x2_vis, y2_vis))

                txt.set_text(
                    f"k={i+1}/{len(steps)}\n[a,b]=[{st.a:.5f},{st.b:.5f}]\n"
                    f"x1={st.x1:.5f}, x2={st.x2:.5f}\n"
                    f"перенос: {st.reuse}"
                )
                writer.grab_frame()

            for _ in range(hfs):
                writer.grab_frame()

    plt.close(fig)


# ============================================================
# 12) Видео: хорды по g(x)=f'(x) (пересечение с y=0 видно каждый шаг)
# ============================================================
def render_chords(steps: List[ChordStep], out_file: Path):
    if not steps:
        return
    fpss, hfs = adaptive_frames(len(steps), FPS, frames_per_step=14, hold_frames=8)

    a0 = steps[0].a
    b0 = steps[0].b
    xlim0 = (a0 - 0.4*(b0-a0), b0 + 0.4*(b0-a0))

    fig, (axf, axg) = plt.subplots(2, 1, figsize=(7.2, 6.8), constrained_layout=True)

    setup_ax_1d(axf, "3) Метод хорд (сверху f(x))")
    f_line = plot_func(axf, f1d_chords, xlim0)

    setup_ax_1d(axg, "Хорда на g(x)=f'(x) пересекает y=0", xlabel="x", ylabel="g(x)")
    axg.axhline(0, color=DARK, linewidth=1.0, alpha=0.6)
    g_line = plot_func(axg, df1d_chords, xlim0)

    v_a = axg.axvline(steps[0].a, color=ITMO_CYAN, linewidth=2.0, alpha=0.9)
    v_b = axg.axvline(steps[0].b, color=ITMO_CYAN, linewidth=2.0, alpha=0.9)

    chord_line, = axg.plot([], [], color=ITMO_ORANGE, linewidth=2.2, alpha=0.95)
    ptg, = axg.plot([], [], "o", color=ITMO_PINK, markersize=6)
    ptf, = axf.plot([], [], "o", color=ITMO_PINK, markersize=6)

    txt = axg.text(0.02, 0.98, "", transform=axg.transAxes, va="top", ha="left",
                   fontsize=10, color=DARK,
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=FPS, bitrate=1800,
        extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    )

    with writer.saving(fig, str(out_file), dpi=DPI):
        for i, st in enumerate(steps):
            progress("chords", i, len(steps))

            zoom_xlim(axf, st.a, st.b, pad_frac=0.35, min_width=0.05)
            zoom_ylim_from_xlim(axf, f1d_chords)
            update_func_line(f_line, f1d_chords, axf.get_xlim())

            axg.set_xlim(axf.get_xlim())
            update_func_line(g_line, df1d_chords, axg.get_xlim())

            xl = axg.get_xlim()
            xs = np.linspace(xl[0], xl[1], 900)
            ys = np.array([df1d_chords(float(x)) for x in xs])
            m = float(np.max(np.abs(ys)))
            m = max(m, 1e-6)
            axg.set_ylim(-1.15*m, 1.15*m)

            prev = steps[i-1] if i > 0 else st

            for j in range(fpss):
                tt = ease(j/(fpss-1) if fpss > 1 else 1.0)

                a_vis = prev.a + (st.a - prev.a)*tt
                b_vis = prev.b + (st.b - prev.b)*tt
                v_a.set_xdata([a_vis, a_vis])
                v_b.set_xdata([b_vis, b_vis])

                ga = df1d_chords(a_vis)
                gb = df1d_chords(b_vis)
                chord_line.set_data([a_vis, b_vis], [ga, gb])

                ptg.set_data([st.x], [df1d_chords(st.x)])
                ptf.set_data([st.x], [f1d_chords(st.x)])

                txt.set_text(
                    f"k={i+1}/{len(steps)}\n"
                    f"[a,b]=[{st.a:.5f},{st.b:.5f}]\n"
                    f"g(a)={df1d_chords(st.a):+.3f}, g(b)={df1d_chords(st.b):+.3f}\n"
                    f"x={st.x:.6f}, g(x)={df1d_chords(st.x):+.3e}"
                )
                writer.grab_frame()

            for _ in range(hfs):
                writer.grab_frame()

    plt.close(fig)


# ============================================================
# 13) Видео: Ньютон (по g=f') — касательная на g
# ============================================================
def render_newton(steps: List[NewtonStep], out_file: Path):
    if not steps:
        return
    fpss, hfs = adaptive_frames(len(steps), FPS, frames_per_step=16, hold_frames=10)

    xs_all = [st.x for st in steps] + [steps[-1].x_next]
    xlim0 = (min(xs_all) - 0.8, max(xs_all) + 0.8)

    fig, (axf, axg) = plt.subplots(2, 1, figsize=(7.2, 6.8), constrained_layout=True)

    setup_ax_1d(axf, "4) Метод Ньютона (сверху f(x))")
    f_line = plot_func(axf, f1d_chords, xlim0)

    setup_ax_1d(axg, "Ньютон на g(x)=f'(x): касательная → пересечение с y=0", xlabel="x", ylabel="g(x)")
    axg.axhline(0, color=DARK, linewidth=1.0, alpha=0.6)
    g_line = plot_func(axg, df1d_chords, xlim0)

    tang, = axg.plot([], [], color=ITMO_ORANGE, linewidth=2.2, alpha=0.95)
    ptg, = axg.plot([], [], "o", color=ITMO_PINK, markersize=6)
    ptf, = axf.plot([], [], "o", color=ITMO_PINK, markersize=6)

    txt = axg.text(0.02, 0.98, "", transform=axg.transAxes, va="top", ha="left",
                   fontsize=10, color=DARK,
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=FPS, bitrate=1800,
        extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    )

    with writer.saving(fig, str(out_file), dpi=DPI):
        for i, st in enumerate(steps):
            progress("newton", i, len(steps))

            axf.set_xlim(*xlim0)
            zoom_ylim_from_xlim(axf, f1d_chords)
            update_func_line(f_line, f1d_chords, axf.get_xlim())

            axg.set_xlim(*xlim0)
            update_func_line(g_line, df1d_chords, axg.get_xlim())

            xl = axg.get_xlim()
            xs = np.linspace(xl[0], xl[1], 900)
            ys = np.array([df1d_chords(float(x)) for x in xs])
            m = float(np.max(np.abs(ys)))
            m = max(m, 1e-6)
            axg.set_ylim(-1.15*m, 1.15*m)

            gx = df1d_chords(st.x)
            gpx = d2f1d_chords(st.x)
            line_x = np.array([st.x - 1.0, st.x + 1.0])
            line_y = gx + gpx * (line_x - st.x)

            for j in range(fpss):
                ptg.set_data([st.x], [gx])
                ptf.set_data([st.x], [f1d_chords(st.x)])
                tang.set_data(line_x, line_y)

                txt.set_text(
                    f"k={i+1}/{len(steps)}\n"
                    f"x={st.x:.6f} → {st.x_next:.6f}\n"
                    f"g(x)={gx:+.3e}, g'(x)={gpx:+.3e}"
                )
                writer.grab_frame()

            for _ in range(hfs):
                writer.grab_frame()

    plt.close(fig)


# ============================================================
# 14) Видео: квадратичная аппроксимация (функция не похожая на параболу)
# ============================================================
def render_quad(steps: List[QuadStep], out_file: Path):
    if not steps:
        return
    fpss, hfs = adaptive_frames(len(steps), FPS, frames_per_step=14, hold_frames=8)

    st0 = steps[0]
    xlim0 = (st0.x1 - 1.0, st0.x3 + 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    setup_ax_1d(ax, "5) Квадратичная аппроксимация (функция не парабола)")
    f_line = plot_func(ax, f1d_quad, xlim0)

    p1, = ax.plot([], [], "o", color=ITMO_CYAN, markersize=6)
    p2, = ax.plot([], [], "o", color=ITMO_CYAN, markersize=6)
    p3, = ax.plot([], [], "o", color=ITMO_CYAN, markersize=6)
    pbar, = ax.plot([], [], "o", color=ITMO_ORANGE, markersize=7)
    pmin, = ax.plot([], [], "o", color=ITMO_PINK, markersize=7)

    parab, = ax.plot([], [], color=ITMO_ORANGE, linewidth=2.0, alpha=0.60, linestyle="--")

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=FPS, bitrate=1800,
        extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    )

    with writer.saving(fig, str(out_file), dpi=DPI):
        for i, st in enumerate(steps):
            progress("quad", i, len(steps))

            zoom_xlim(ax, st.x1, st.x3, pad_frac=0.55, min_width=0.08)
            zoom_ylim_from_xlim(ax, f1d_quad)
            update_func_line(f_line, f1d_quad, ax.get_xlim())

            xpts = np.array([st.x1, st.x2, st.x3], dtype=float)
            ypts = np.array([st.f1, st.f2, st.f3], dtype=float)
            A = np.vstack([xpts**2, xpts, np.ones_like(xpts)]).T
            try:
                a2, a1, a0c = np.linalg.solve(A, ypts)
                xsq = np.linspace(min(xpts)-0.4, max(xpts)+0.4, 220)
                ysq = a2*xsq**2 + a1*xsq + a0c
                parab.set_data(xsq, ysq)
            except np.linalg.LinAlgError:
                parab.set_data([], [])

            for j in range(fpss):
                p1.set_data([st.x1], [st.f1])
                p2.set_data([st.x2], [st.f2])
                p3.set_data([st.x3], [st.f3])
                pbar.set_data([st.xbar], [st.fbar])
                pmin.set_data([st.xmin], [st.fmin])

                txt.set_text(
                    f"k={i+1}/{len(steps)}\n"
                    f"(x1,x2,x3)=({st.x1:.3f},{st.x2:.3f},{st.x3:.3f})\n"
                    f"x̄={st.xbar:.5f}\n"
                    f"x*={st.xmin:.5f}, f*={st.fmin:.5f}"
                )
                writer.grab_frame()

            for _ in range(hfs):
                writer.grab_frame()

    plt.close(fig)


# ============================================================
# 15) Видео: 2D пути (касательная/нормаль)
# ============================================================
def set_single_quiver(qv, x, y, u, v):
    qv.set_offsets(np.array([[x, y]]))
    qv.set_UVC(np.array([u]), np.array([v]))

def render_path2d(path: Path2D, title: str, out_file: Path, zoom: bool = True, zoom_recent: int = 6):
    if not path.points:
        return

    pts_all = np.array(path.points)
    xmin, ymin = pts_all.min(axis=0) - 1.2
    xmax, ymax = pts_all.max(axis=0) + 1.2

    xx = np.linspace(xmin, xmax, 200)
    yy = np.linspace(ymin, ymax, 200)
    X, Y = np.meshgrid(xx, yy)
    Z = f2d(X, Y)

    fig, ax = plt.subplots(figsize=(6.9, 5.6), constrained_layout=True)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title, color=DARK, fontsize=12, fontweight="bold")

    levels = np.linspace(np.min(Z), np.max(Z), 18)
    ax.contour(X, Y, Z, levels=levels, linewidths=1.0, alpha=0.55)

    line, = ax.plot([], [], color=ITMO_BLUE, linewidth=2.4, alpha=0.9)
    pt, = ax.plot([], [], "o", color=ITMO_PINK, markersize=7)
    step_line, = ax.plot([], [], color=ITMO_ORANGE, linewidth=2.0, alpha=0.85)

    normal = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1,
                       color=ITMO_ORANGE, width=0.013, zorder=6)
    tang = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1,
                     color=ITMO_LIME, width=0.013, zorder=6)

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=FPS, bitrate=1800,
        extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"]
    )

    frames_per_step = 12
    hold_frames = 6
    frames_per_step, hold_frames = adaptive_frames(len(path.points), FPS, frames_per_step, hold_frames)

    with writer.saving(fig, str(out_file), dpi=DPI):
        for i in range(len(path.points)):
            if i > 0:
                progress(out_file.stem, i-1, len(path.points)-1)

            p_prev = path.points[i-1] if i > 0 else path.points[i]
            p_cur = path.points[i]
            g_cur = path.grads[i]

            if zoom and i > 0:
                recent = np.array(path.points[max(0, i-zoom_recent):i+1])
                xmn, ymn = recent.min(axis=0)
                xmx, ymx = recent.max(axis=0)
                wx = max(1.2, float(xmx - xmn) * 1.8)
                wy = max(1.2, float(ymx - ymn) * 1.8)
                cx = float(recent[-1, 0]); cy = float(recent[-1, 1])
                ax.set_xlim(cx - wx/2, cx + wx/2)
                ax.set_ylim(cy - wy/2, cy + wy/2)

            for j in range(frames_per_step):
                t = j/(frames_per_step-1) if frames_per_step > 1 else 1.0
                tt = ease(t)

                x = p_prev[0] + (p_cur[0] - p_prev[0]) * tt
                y = p_prev[1] + (p_cur[1] - p_prev[1]) * tt

                pts = np.array(path.points[:i+1])
                line.set_data(pts[:, 0], pts[:, 1])
                pt.set_data([x], [y])

                if i > 0:
                    step_line.set_data([p_prev[0], p_cur[0]], [p_prev[1], p_cur[1]])
                else:
                    step_line.set_data([], [])

                ng = norm2(g_cur)
                if ng < 1e-12:
                    u1=v1=u2=v2=0.0
                else:
                    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
                    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
                    scale = 0.28 * float(min(xspan, yspan))
                    u1, v1 = (-g_cur[0]/ng)*scale, (-g_cur[1]/ng)*scale
                    u2, v2 = (-v1, u1)

                set_single_quiver(normal, x, y, u1, v1)
                set_single_quiver(tang, x, y, u2, v2)

                info = path.step_info[i] if i < len(path.step_info) else ""
                txt.set_text(
                    f"шаг {i}/{len(path.points)-1}\n"
                    f"f={float(f2d(np.array([x]), np.array([y]))[0]):.5f}\n"
                    f"||grad||={ng:.5f}\n"
                    f"{info}"
                )
                writer.grab_frame()

            for _ in range(hold_frames):
                writer.grab_frame()

    plt.close(fig)


# ============================================================
# 16) Склейка: перекодировать и привести к одному размеру
# ============================================================
def concat_videos_reencode(files: List[Path], out_file: Path, target_w: int = 1280, target_h: int = 720):
    """
    Склейка через concat demuxer + перекодирование.
    Это устойчиво к разным разрешениям входных роликов.
    """
    ff = ffmpeg_bin()

    list_path = out_file.with_suffix(".concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            f.write("file '" + p.resolve().as_posix() + "'\n")

    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [
        ff, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(out_file)
    ]

    subprocess.run(cmd, check=True)
    try:
        list_path.unlink()
    except Exception:
        pass


# ============================================================
# 17) main
# ============================================================
def main():
    ensure_ffmpeg()

    print("\n=== 1/8: половинное деление ===")
    s1 = halving_steps(A1D, B1D, EPS_1D)

    print("\n=== 2/8: золотое сечение ===")
    s2 = golden_steps(A1D, B1D, EPS_1D)

    print("\n=== 3/8: хорды по производной ===")
    s3 = chords_on_derivative(A1D, B1D, EPS_1D)

    print("\n=== 4/8: Ньютон ===")
    s4 = newton_on_derivative(X0_NEWTON, EPS_1D, min_steps=6)

    print("\n=== 5/8: квадратичная аппроксимация ===")
    s5 = quad_steps(A1D, B1D, X1_QUAD, H_QUAD, EPS_1D, min_steps=10)

    print("\n=== 6/8: покоординатный спуск ===")
    p6 = coord_descent_2d(X0_2D, EPS_2D, MAX_IT_2D)

    print("\n=== 7/8: градиентный спуск ===")
    p7 = gradient_descent_2d(X0_2D, EPS_2D, alpha=0.10, max_iter=MAX_IT_2D)

    print("\n=== 8/8: наискорейший спуск ===")
    p8 = steepest_descent_2d(X0_2D, EPS_2D, max_iter=MAX_IT_2D)

    videos: List[Path] = []

    v1 = OUT_DIR / "01_halving.mp4"
    render_halving(s1, v1)
    videos.append(v1)

    v2 = OUT_DIR / "02_golden.mp4"
    render_golden(s2, v2)
    videos.append(v2)

    v3 = OUT_DIR / "03_chords.mp4"
    render_chords(s3, v3)
    videos.append(v3)

    v4 = OUT_DIR / "04_newton.mp4"
    render_newton(s4, v4)
    videos.append(v4)

    v5 = OUT_DIR / "05_quad.mp4"
    render_quad(s5, v5)
    videos.append(v5)

    v6 = OUT_DIR / "06_coord.mp4"
    render_path2d(p6, "6) Покоординатный спуск", v6, zoom=True)
    videos.append(v6)

    v7 = OUT_DIR / "07_grad.mp4"
    render_path2d(p7, "7) Градиентный спуск (нормаль и касательная)", v7, zoom=True)
    videos.append(v7)

    v8 = OUT_DIR / "08_steepest.mp4"
    render_path2d(p8, "8) Наискорейший спуск (нормаль и касательная)", v8, zoom=True)
    videos.append(v8)

    combined = OUT_DIR / "combined.mp4"
    print("\n=== Склейка (re-encode + resize 1280x720) ===")
    concat_videos_reencode(videos, combined, target_w=1280, target_h=720)
    print("Готово:", combined)

if __name__ == "__main__":
    main()