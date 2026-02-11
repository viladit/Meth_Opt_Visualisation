# -*- coding: utf-8 -*-
"""
ITMO / Методы оптимизации — визуализация (mp4), v2 (наглядно)

Фиксы:
- Newton/Quadratic: шаги растянуты на кадры + hold, плюс умный зум (видно движение).
- 1D-методы: умный зум по текущему [a,b] (иначе кажется, что "стоит").
- Quadratic: по умолчанию DEMO-функция (иначе на "лабораторной" может сойтись за 1 шаг).
- Склейка mp4 последовательно (не плиткой) через твой FFMPEG_PATH.

Запуск:
    pip install numpy matplotlib
    python itmo_opt_viz_all_methods_v2.py

Результат:
    videos_itmo_v2/
        01_dichotomy.mp4
        ...
        08_steepest_descent.mp4
        combined_sequential.mp4
"""

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
# 0) Цвета (поиграйся здесь, чтобы было "авторски")
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
# 1) FFmpeg (Windows)
# ============================================================
FFMPEG_PATH: Optional[str] = r"/Users/viladit/PycharmProjects/MethOpt/ffmpeg"

def ensure_ffmpeg():
    if FFMPEG_PATH:
        p = Path(FFMPEG_PATH)
        if not p.exists():
            raise RuntimeError(f"FFMPEG_PATH указан, но файла нет: {FFMPEG_PATH}")
        mpl.rcParams["animation.ffmpeg_path"] = str(p)
        return
    # fallback
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg не найден: укажи FFMPEG_PATH или добавь ffmpeg в PATH.")

def ffmpeg_cmd() -> str:
    return str(mpl.rcParams.get("animation.ffmpeg_path") or "ffmpeg")


# ============================================================
# 2) Утилиты
# ============================================================
def _ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)

def bracket_minimum(phi: Callable[[float], float], step: float = 1.0, grow: float = 2.0,
                    max_expand: int = 60) -> Tuple[float, float]:
    """Находит отрезок [L,R], на котором гарантированно лежит минимум phi(λ).
    Важно: умеет искать и в отрицательную сторону (для 2D методов это критично),
    иначе можно получить λ≈0 и ощущение, что метод 'стоит на месте'.
    """
    lam0 = 0.0
    f0 = phi(lam0)

    # пробуем обе стороны
    lam_pos = abs(step)
    lam_neg = -abs(step)
    f_pos = phi(lam_pos)
    f_neg = phi(lam_neg)

    # выбираем направление, где есть убывание
    if f_pos >= f0 and f_neg >= f0:
        # минимума рядом не нашли — возвращаем маленький интервал вокруг 0
        return (lam_neg, lam_pos)

    if f_pos <= f_neg:
        direction = +1.0
        lam1, f1 = lam_pos, f_pos
    else:
        direction = -1.0
        lam1, f1 = lam_neg, f_neg

    # расширяемся в выбранном направлении
    prev_lam, prev_f = lam0, f0
    for _ in range(max_expand):
        lam2 = lam1 * grow
        f2 = phi(lam2)
        if f2 >= f1:
            L, R = (prev_lam, lam2) if prev_lam < lam2 else (lam2, prev_lam)
            return L, R
        prev_lam, prev_f = lam1, f1
        lam1, f1 = lam2, f2

    # fallback
    L, R = (lam0, lam1) if lam0 < lam1 else (lam1, lam0)
    return L, R

def golden_section_min(phi: Callable[[float], float], a: float, b: float, tol: float = 1e-7,
                       max_iter: int = 100000) -> float:
    gr = (math.sqrt(5) - 1) / 2
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1 = phi(x1)
    f2 = phi(x2)
    for _ in range(max_iter):
        if abs(b - a) <= tol:
            break
        if f1 > f2:
            a = x1
            x1 = x2; f1 = f2
            x2 = a + gr * (b - a); f2 = phi(x2)
        else:
            b = x2
            x2 = x1; f2 = f1
            x1 = b - gr * (b - a); f1 = phi(x1)
    return (a + b) / 2

def norm2(v: Tuple[float, float]) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

def setup_ax_1d(ax: plt.Axes, title: str):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title, color=DARK, fontsize=12, fontweight="bold")

def draw_function_1d(ax: plt.Axes, xlim: Tuple[float, float]):
    """Создаёт линию f(x) и возвращает Line2D (чтобы можно было обновлять при зуме)."""
    line, = ax.plot([], [], color=ITMO_BLUE, linewidth=2.2, alpha=0.92)
    update_function_line(line, xlim)
    return line

def update_function_line(line, xlim: Tuple[float, float], n: int = 1400):
    """Обновляет дискретизацию кривой под текущий xlim.
    Это убирает эффект, когда при сильном зуме кривая превращается в 'ломаную', а точки будто бы не на ней.
    """
    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(n))
    ys = np.array([f1d(float(x)) for x in xs])
    line.set_data(xs, ys)

def zoom_xlim(ax: plt.Axes, L: float, R: float, pad_frac: float = 0.30, min_width: float = 0.05):
    w = max(abs(R - L), min_width)
    pad = w * pad_frac
    ax.set_xlim(L - pad, R + pad)

def zoom_ylim_from_xlim(ax: plt.Axes):
    x0, x1 = ax.get_xlim()
    xs = np.linspace(x0, x1, 500)
    ys = np.array([f1d(float(x)) for x in xs])
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    pad = max(1e-9, (y_max - y_min) * 0.18)
    ax.set_ylim(y_min - pad, y_max + pad)


# ============================================================
# 2.1) Производительность / прогресс
# ============================================================
def _adaptive_frames(n_steps: int, fps: int, frames_per_step: int, hold_frames: int,
                     target_seconds: float = 22.0, min_frames_per_step: int = 2,
                     min_hold_frames: int = 0) -> Tuple[int, int]:
    """
    Делает видео "вменяемой" длины и ускоряет рендер на длинных методах.
    - Если шагов мало — оставляет красивую плавность.
    - Если шагов много — урезает кадры/холды так, чтобы уложиться примерно в target_seconds.
    """
    if n_steps <= 0:
        return frames_per_step, hold_frames
    target_total_frames = max(30, int(fps * target_seconds))
    # сколько кадров можем позволить на 1 шаг
    budget_per_step = max(min_frames_per_step, target_total_frames // n_steps)
    new_fps = max(min_frames_per_step, min(frames_per_step, budget_per_step))
    # холд делаем не больше 1/2 от шага и часто убираем вовсе
    new_hold = min(hold_frames, max(min_hold_frames, budget_per_step // 3))
    return new_fps, new_hold

def _zoom_include_points(ax: plt.Axes, xs: List[float], pad_frac: float = 0.30, min_width: float = 0.05):
    L = float(min(xs)); R = float(max(xs))
    zoom_xlim(ax, L, R, pad_frac=pad_frac, min_width=min_width)

def _progress(prefix: str, i: int, n: int):
    """Прогресс без '1 шаг — сразу 100%'.
    - Для коротких прогонов печатаем каждый шаг.
    - Для длинных — каждые ~5% + старт/финиш.
    """
    if n <= 0:
        return
    if n <= 25:
        pct = int(round(100 * (i + 1) / n))
        print(f"[{prefix}] {i+1}/{n} ({pct}%)", flush=True)
        return
    # длинные: ~каждые 5%
    every = max(1, n // 20)
    if i == 0 or i == n - 1 or (i % every == 0):
        pct = int(round(100 * (i + 1) / n))
        print(f"[{prefix}] {i+1}/{n} ({pct}%)", flush=True)


# ============================================================
# 3) 1D функции: LAB vs DEMO (для наглядности)
# ============================================================
# ВАЖНО: чтобы Newton/Quadratic визуально "не стояли", включи DEMO.
# Для "как в лабе" — поставь False.
# --- DEMO-переключатели для наглядности в видео ---
USE_DEMO_1D_FOR_VIDEO = True   # базовая 1D-функция для 1D-методов
USE_DEMO_CHORDS_FUNC  = True   # отдельная "сложная" 1D-функция для метода хорд/Ньютона (по g=f')
USE_DEMO_QUAD_FUNC    = True   # отдельная 1D-функция для квадр. аппроксимации


def f1d_lab(x: float) -> float:
    return (1.0/3.0)*x**3 - 5.0*x + x*math.log(x)

def df1d_lab(x: float) -> float:
    return x**2 + math.log(x) - 4.0

def d2f1d_lab(x: float) -> float:
    return 2.0*x + 1.0/x

def f1d_demo(x: float) -> float:
    return (x - 2.3)**4 + 0.2*(x - 2.3)**2 + 0.15*x

def df1d_demo(x: float) -> float:
    return 4*(x - 2.3)**3 + 0.4*(x - 2.3) + 0.15

def d2f1d_demo(x: float) -> float:
    return 12*(x - 2.3)**2 + 0.4

def f1d_chord_demo(x: float) -> float:
    # Более "сложная" унимодальная функция, чтобы хорды на g(x)=f'(x) были наглядными
    return (x - 2.0)**4 + 0.15*(x - 2.0)**2 + 0.10*math.sin(3.0*x) + 0.12*x

def df1d_chord_demo(x: float) -> float:
    return 4*(x - 2.0)**3 + 0.30*(x - 2.0) + 0.30*math.cos(3.0*x) + 0.12

def d2f1d_chord_demo(x: float) -> float:
    return 12*(x - 2.0)**2 + 0.30 - 0.90*math.sin(3.0*x)

def f1d_quad_demo(x: float) -> float:
    # Функция "не похожая на параболу": квартика + небольшая синусоида (но унимодальна на рабочем интервале)
    return (x - 2.3)**4 + 0.2*(x - 2.3)**2 + 0.15*x + 0.08*math.sin(4.0*x)



def f1d(x: float) -> float:
    return f1d_demo(x) if USE_DEMO_1D_FOR_VIDEO else f1d_lab(x)

def df1d(x: float) -> float:
    return df1d_demo(x) if USE_DEMO_1D_FOR_VIDEO else df1d_lab(x)

def d2f1d(x: float) -> float:
    return d2f1d_demo(x) if USE_DEMO_1D_FOR_VIDEO else d2f1d_lab(x)


# ============================================================
# 4) 2D (ЛР4)
# ============================================================
def f2d(x1: float, x2: float) -> float:
    return 7*x1*x1 + 3*x2*x2 + 0.5*x1*x2 - 3*x1 - 5*x2 + 2

def grad2d(x: Tuple[float, float]) -> Tuple[float, float]:
    x1, x2 = x
    return (14*x1 + 0.5*x2 - 3, 6*x2 + 0.5*x1 - 5)


# ============================================================
# 5) Шаги (dataclass)
# ============================================================
@dataclass
class DichotomyStep:
    a: float; b: float
    x1: float; x2: float
    y1: float; y2: float
    chosen: str

@dataclass
class GoldenStep:
    a: float; b: float
    x1: float; x2: float
    y1: float; y2: float
    chosen: str

@dataclass
class ChordStep:
    a: float; b: float
    x: float
    ga: float; gb: float; gx: float

@dataclass
class NewtonStep:
    x: float
    gx: float
    gpx: float
    x_next: float

@dataclass
class QuadApproxStep:
    x1: float; x2: float; x3: float
    xbar: float
    f1: float; f2: float; f3: float
    fbar: float
    xmin: float; fmin: float
    inside: bool

@dataclass
class Path2D:
    points: List[Tuple[float, float]]
    grads: List[Tuple[float, float]]
    info: List[str]


# ============================================================
# 6) 1D методы
# ============================================================
def dichotomy_min(a0: float, b0: float, eps: float, max_iter: int = 100000) -> List[DichotomyStep]:
    a, b = a0, b0
    steps: List[DichotomyStep] = []
    for _ in range(max_iter):
        x1 = (a + b - eps) / 2.0
        x2 = (a + b + eps) / 2.0
        y1 = f1d(x1); y2 = f1d(x2)
        if y1 > y2:
            chosen = "a=x1"; a = x1
        else:
            chosen = "b=x2"; b = x2
        steps.append(DichotomyStep(a=a, b=b, x1=x1, x2=x2, y1=y1, y2=y2, chosen=chosen))
        if (b - a) <= 2.0 * eps:
            break
    return steps

def golden_min(a0: float, b0: float, eps: float, max_iter: int = 100000) -> List[GoldenStep]:
    gr = (math.sqrt(5) - 1) / 2
    a, b = a0, b0
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    y1 = f1d(x1); y2 = f1d(x2)
    steps: List[GoldenStep] = []
    for _ in range(max_iter):
        if y1 > y2:
            chosen = "a=x1"
            a = x1
            x1, y1 = x2, y2
            x2 = a + gr * (b - a); y2 = f1d(x2)
        else:
            chosen = "b=x2"
            b = x2
            x2, y2 = x1, y1
            x1 = b - gr * (b - a); y1 = f1d(x1)
        steps.append(GoldenStep(a=a, b=b, x1=x1, x2=x2, y1=y1, y2=y2, chosen=chosen))
        if (b - a) <= eps:
            break
    return steps

def chords_on_derivative(a0: float, b0: float, eps: float, max_iter: int = 100000) -> List[ChordStep]:
    # g(x)=f'(x). Для наглядности можно включить отдельную демо-функцию.
    g = df1d_chord_demo if USE_DEMO_CHORDS_FUNC else df1d

    a, b = a0, b0
    ga = g(a); gb = g(b)
    if ga * gb > 0:
        raise ValueError("Для метода хорд нужно, чтобы g(a)=f'(a) и g(b)=f'(b) имели разные знаки.")
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
    # Ищем корень g(x)=f'(x)=0 методом Ньютона. Для наглядности можно включить демо-функцию.
    g = df1d_chord_demo if USE_DEMO_CHORDS_FUNC else df1d
    gp = d2f1d_chord_demo if USE_DEMO_CHORDS_FUNC else d2f1d

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

def quad_approx_min(a: float, b: float, x1: float, h: float, eps: float,
                    min_steps: int = 10, max_iter: int = 5000, func: Callable[[float], float] = f1d) -> List[QuadApproxStep]:
    """Квадратичная аппроксимация (параболическая интерполяция) без 'залипания'.

    Проблема предыдущей версии: после шага, где xmin=старый x2, мы снова
    восстанавливали ту же тройку через (x1, h) => (x1, x2=x1+h, x3=...) и
    получали *точно те же точки*, поэтому на видео казалось, что метод завис.

    Здесь мы храним тройку (x1<x2<x3) явно и на каждом шаге выбираем новую тройку
    вокруг лучшей точки так, чтобы она реально менялась.
    """

    def clamp(x: float) -> float:
        return max(a, min(b, x))

    # стартовая тройка
    x2 = clamp(x1)
    h = float(h) if abs(h) > 0 else 0.1
    x1 = clamp(x2 - abs(h))
    x3 = clamp(x2 + abs(h))
    if x1 == x2:
        x1 = clamp(x2 - 2*abs(h))
    if x3 == x2:
        x3 = clamp(x2 + 2*abs(h))
    if not (x1 < x2 < x3):
        # fallback: расставим равномерно
        x1, x2, x3 = sorted([x1, x2, x3])
        if x1 == x2:
            x2 = clamp((x1 + x3) / 2)

    steps: List[QuadApproxStep] = []
    prev_best: Optional[float] = None

    for _ in range(max_iter):
        f1 = func(x1); f2 = func(x2); f3 = func(x3)

        # xbar по формуле параболической интерполяции
        denom = (x2 - x3)*f1 + (x3 - x1)*f2 + (x1 - x2)*f3
        numer = (x2**2 - x3**2)*f1 + (x3**2 - x1**2)*f2 + (x1**2 - x2**2)*f3
        if abs(denom) < 1e-14:
            xbar = x2
        else:
            xbar = 0.5 * numer / denom
        xbar = clamp(xbar)
        fbar = func(xbar)

        candidates = [(x1, f1), (x2, f2), (x3, f3), (xbar, fbar)]
        xmin, fmin = min(candidates, key=lambda t: t[1])
        inside = (x1 <= xbar <= x3)

        steps.append(QuadApproxStep(x1=x1, x2=x2, x3=x3,
                                   xbar=xbar, f1=f1, f2=f2, f3=f3,
                                   fbar=fbar, xmin=xmin, fmin=fmin, inside=inside))

        # стоп: как ты хочешь — по изменению лучшей точки (и ширине интервала)
        if len(steps) >= min_steps:
            if prev_best is not None and abs(xmin - prev_best) <= eps:
                break
            if abs(x3 - x1) <= eps:
                break
        prev_best = xmin

        # --- обновление тройки вокруг xmin ---
        pts = sorted(set([round(x, 15) for x, _ in candidates]))
        # восстановим значения (без потери на round)
        pts = sorted([x1, x2, x3, xbar])
        # найдём ближайшие слева/справа от xmin
        lefts = [x for x in pts if x < xmin - 1e-15]
        rights = [x for x in pts if x > xmin + 1e-15]

        if lefts and rights:
            x2 = xmin
            x1 = max(lefts)
            x3 = min(rights)
        else:
            # если нет одной стороны — уменьшаем шаг и строим симметрично вокруг xmin
            span = max(1e-6, abs(x3 - x1) * 0.5)
            span = max(span * 0.6, eps * 5)
            x2 = xmin
            x1 = clamp(x2 - span)
            x3 = clamp(x2 + span)

        # гарантируем строгий порядок (иначе polyfit/рисование будет вести себя странно)
        if not (x1 < x2 < x3):
            x1, x2, x3 = sorted([x1, x2, x3])
            if x1 == x2:
                x2 = clamp((x1 + x3) / 2)

    return steps


# ============================================================
# 7) 2D методы
# ============================================================
def coord_descent_2d(x0: Tuple[float, float], eps: float, min_cycles_for_video: int = 12,
                     max_cycles: int = 100000) -> Path2D:
    x = (float(x0[0]), float(x0[1]))
    points = [x]
    grads = [grad2d(x)]
    info  = ["start"]

    def phi_x1(lam: float, x_fixed: Tuple[float, float]) -> float:
        return f2d(x_fixed[0] + lam, x_fixed[1])

    def phi_x2(lam: float, x_fixed: Tuple[float, float]) -> float:
        return f2d(x_fixed[0], x_fixed[1] + lam)

    for cyc in range(max_cycles):
        x_prev = x

        # x1
        phi = lambda lam: phi_x1(lam, x)
        L, R = bracket_minimum(phi, step=1.0)
        lam = golden_section_min(phi, L, R, tol=1e-7)
        x = (x[0] + lam, x[1])
        points.append(x); grads.append(grad2d(x)); info.append(f"update x1 (λ={lam:.3g})")

        # x2
        phi = lambda lam: phi_x2(lam, x)
        L, R = bracket_minimum(phi, step=1.0)
        lam = golden_section_min(phi, L, R, tol=1e-7)
        x = (x[0], x[1] + lam)
        points.append(x); grads.append(grad2d(x)); info.append(f"update x2 (λ={lam:.3g})")

        # Остановка как ты просишь: ||x_n - x_{n-1}|| < eps (после полного цикла по координатам)
        # Градиентный критерий тут не используем, иначе может остановить слишком рано и будет казаться,
        # что метод "не едет".
        done = (norm2((x[0]-x_prev[0], x[1]-x_prev[1])) <= eps)
        if done and cyc >= min_cycles_for_video:
            break

    return Path2D(points=points, grads=grads, info=info)

def gradient_descent_2d(x0: Tuple[float, float], eps: float, h0: float = 0.25,
                        max_iter: int = 100000) -> Path2D:
    x = (float(x0[0]), float(x0[1]))
    points = [x]
    grads  = [grad2d(x)]
    info   = [f"start (h={h0})"]
    h = h0

    for _ in range(max_iter):
        g = grad2d(x)
        if norm2(g) <= eps:
            break
        fx = f2d(*x)
        tries = 0
        while True:
            x_new = (x[0] - h*g[0], x[1] - h*g[1])
            if f2d(*x_new) <= fx:
                x = x_new
                points.append(x); grads.append(grad2d(x)); info.append(f"h={h:.3g}, tries={tries}")
                break
            h *= 0.5
            tries += 1
            if tries > 40:
                x = x_new
                points.append(x); grads.append(grad2d(x)); info.append(f"h={h:.3g}, tries={tries} (forced)")
                break
    return Path2D(points=points, grads=grads, info=info)

def steepest_descent_2d(x0: Tuple[float, float], eps: float, max_iter: int = 100000) -> Path2D:
    x = (float(x0[0]), float(x0[1]))
    points = [x]
    grads  = [grad2d(x)]
    info   = ["start"]

    for _ in range(max_iter):
        g = grad2d(x)
        ng = norm2(g)
        if ng <= eps:
            break
        S = (-g[0]/ng, -g[1]/ng)
        phi = lambda lam: f2d(x[0] + lam*S[0], x[1] + lam*S[1])
        L, R = bracket_minimum(phi, step=1.0)
        lam = golden_section_min(phi, L, R, tol=1e-7)
        x = (x[0] + lam*S[0], x[1] + lam*S[1])
        points.append(x); grads.append(grad2d(x)); info.append(f"λ={lam:.3g}")

    return Path2D(points=points, grads=grads, info=info)


# ============================================================
# 8) Рендер видео 1D (с зумом и растягиванием шагов)
# ============================================================
def render_dichotomy_video(steps: List[DichotomyStep], out_file: Path, fps: int, dpi: int,
                           frames_per_step: int = 10, hold_frames: int = 6):
    if not steps:
        return
    frames_per_step, hold_frames = _adaptive_frames(len(steps), fps, frames_per_step, hold_frames)
    a0 = steps[0].a; b0 = steps[0].b
    xlim0 = (a0 - 0.4*(b0-a0), b0 + 0.4*(b0-a0))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    setup_ax_1d(ax, "1) Дихотомия")
    f_line = draw_function_1d(ax, xlim0)

    v_a = ax.axvline(steps[0].a, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)
    v_b = ax.axvline(steps[0].b, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)
    p1, = ax.plot([], [], "o", color=ITMO_PINK, markersize=6)
    p2, = ax.plot([], [], "o", color=ITMO_PINK, markersize=6)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(fps=fps, metadata={"title": out_file.stem}, bitrate=1800, extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"])
    with writer.saving(fig, str(out_file), dpi=dpi):
        for i, st in enumerate(steps):
            _progress("dichotomy", i, len(steps))
            _zoom_include_points(ax, [st.a, st.b, st.x1, st.x2], pad_frac=0.35, min_width=0.02)
            zoom_ylim_from_xlim(ax)
            update_function_line(f_line, ax.get_xlim())
            update_function_line(f_line, ax.get_xlim())
            update_function_line(f_line, ax.get_xlim())
            for j in range(frames_per_step):
                t = j/(frames_per_step-1) if frames_per_step > 1 else 1.0
                tt = _ease(t)
                a_prev = steps[i-1].a if i > 0 else st.a
                b_prev = steps[i-1].b if i > 0 else st.b
                a_vis = a_prev + (st.a - a_prev) * tt
                b_vis = b_prev + (st.b - b_prev) * tt
                v_a.set_xdata([a_vis, a_vis])
                v_b.set_xdata([b_vis, b_vis])
                p1.set_data([st.x1], [st.y1])
                p2.set_data([st.x2], [st.y2])
                txt.set_text(
                    f"k={i+1}/{len(steps)}\n[a,b]=[{st.a:.5f},{st.b:.5f}]\n"
                    f"x1={st.x1:.5f}, x2={st.x2:.5f}\n"
                    f"f1={st.y1:.5f}, f2={st.y2:.5f}\nupdate: {st.chosen}"
                )
                writer.grab_frame()
            for _ in range(hold_frames):
                writer.grab_frame()
    plt.close(fig)

def render_golden_video(steps: List[GoldenStep], out_file: Path, fps: int, dpi: int,
                        frames_per_step: int = 10, hold_frames: int = 6):
    if not steps:
        return
    frames_per_step, hold_frames = _adaptive_frames(len(steps), fps, frames_per_step, hold_frames)

    a0 = steps[0].a
    b0 = steps[0].b
    xlim0 = (a0 - 0.4*(b0-a0), b0 + 0.4*(b0-a0))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    setup_ax_1d(ax, "2) Золотое сечение (видно перенос точки)")
    f_line = draw_function_1d(ax, xlim0)

    v_a = ax.axvline(steps[0].a, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)
    v_b = ax.axvline(steps[0].b, color=ITMO_CYAN, linewidth=2.2, alpha=0.9)

    # Роли (x1 и x2). На каждом шаге одна роль переиспользуется — это будет видно.
    p1, = ax.plot([], [], "o", color=ITMO_PINK, markersize=7)    # x1
    p2, = ax.plot([], [], "o", color=ITMO_ORANGE, markersize=7)  # x2
    t1 = ax.text(0, 0, "x1", color=DARK, fontsize=10, ha="left", va="bottom")
    t2 = ax.text(0, 0, "x2", color=DARK, fontsize=10, ha="left", va="bottom")

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": out_file.stem},
        bitrate=1800,
        extra_args=["-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"],
    )

    with writer.saving(fig, str(out_file), dpi=dpi):
        for i, st in enumerate(steps):
            _progress("golden", i, len(steps))

            _zoom_include_points(ax, [st.a, st.b, st.x1, st.x2], pad_frac=0.35, min_width=0.02)
            zoom_ylim_from_xlim(ax)
            update_function_line(f_line, ax.get_xlim())

            prev = steps[i-1] if i > 0 else st

            for j in range(frames_per_step):
                t = j/(frames_per_step-1) if frames_per_step > 1 else 1.0
                tt = _ease(t)

                a_vis = prev.a + (st.a - prev.a) * tt
                b_vis = prev.b + (st.b - prev.b) * tt
                v_a.set_xdata([a_vis, a_vis])
                v_b.set_xdata([b_vis, b_vis])

                x1_vis = prev.x1 + (st.x1 - prev.x1) * tt
                x2_vis = prev.x2 + (st.x2 - prev.x2) * tt
                y1_vis = f1d(x1_vis)
                y2_vis = f1d(x2_vis)

                p1.set_data([x1_vis], [y1_vis])
                p2.set_data([x2_vis], [y2_vis])
                t1.set_position((x1_vis, y1_vis))
                t2.set_position((x2_vis, y2_vis))

                txt.set_text(
                    f"k={i+1}/{len(steps)}\n[a,b]=[{st.a:.5f},{st.b:.5f}]\n"
                    f"x1={st.x1:.5f}, x2={st.x2:.5f}\n"
                    f"перенос: {st.chosen} (одна точка переиспользуется)"
                )

                writer.grab_frame()

            for _ in range(hold_frames):
                writer.grab_frame()

    plt.close(fig)


def render_chords_video(steps: List[ChordStep], out_file: Path, fps: int, dpi: int,
                        frames_per_step: int = 14, hold_frames: int = 8):
    if not steps:
        return
    frames_per_step, hold_frames = _adaptive_frames(len(steps), fps, frames_per_step, hold_frames)

    # Для наглядности хорды рисуем на более "сложной" функции (по просьбе преподавателя)
    f_vis = f1d_chord_demo if USE_DEMO_CHORDS_FUNC else f1d
    g_vis = df1d_chord_demo if USE_DEMO_CHORDS_FUNC else df1d

    a0 = steps[0].a
    b0 = steps[0].b
    xlim0 = (a0 - 0.4*(b0-a0), b0 + 0.4*(b0-a0))

    fig, (axf, axg) = plt.subplots(2, 1, figsize=(7.2, 6.8), constrained_layout=True)
    setup_ax_1d(axf, "3) Метод хорд по g(x)=f'(x) — сверху f(x)")
    f_line = draw_function_1d(axf, xlim0)
    xs0 = np.linspace(xlim0[0], xlim0[1], 800)
    f_line.set_data(xs0, np.array([f_vis(float(x)) for x in xs0]))

    axg.set_facecolor(BG)
    axg.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    axg.set_xlabel("x")
    axg.set_ylabel("g(x)=f'(x)")
    axg.set_title("Хорда на g(x): пересечение с y=0 видно на каждом шаге", color=DARK, fontsize=11, fontweight="bold")
    axg.axhline(0, color=DARK, linewidth=1.0, alpha=0.55)

    xs = np.linspace(xlim0[0], xlim0[1], 900)
    gs = np.array([g_vis(float(x)) for x in xs])
    g_curve, = axg.plot(xs, gs, color=ITMO_BLUE, linewidth=2.2, alpha=0.9)

    v_a = axg.axvline(steps[0].a, color=ITMO_CYAN, linewidth=2.0, alpha=0.9)
    v_b = axg.axvline(steps[0].b, color=ITMO_CYAN, linewidth=2.0, alpha=0.9)
    chord_line, = axg.plot([], [], color=ITMO_ORANGE, linewidth=2.2, alpha=0.95)
    pt, = axg.plot([], [], "o", color=ITMO_PINK, markersize=6)
    pt_f, = axf.plot([], [], "o", color=ITMO_PINK, markersize=6)

    txt = axg.text(0.02, 0.98, "", transform=axg.transAxes, va="top", ha="left",
                   fontsize=10, color=DARK,
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": out_file.stem},
        bitrate=1800,
        extra_args=["-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"],
    )

    with writer.saving(fig, str(out_file), dpi=dpi):
        for i, st in enumerate(steps):
            _progress("chords", i, len(steps))

            zoom_xlim(axf, st.a, st.b, pad_frac=0.35, min_width=0.02)
            zoom_ylim_from_xlim(axf)
            xsf = np.linspace(axf.get_xlim()[0], axf.get_xlim()[1], 800)
            f_line.set_data(xsf, np.array([f_vis(float(x)) for x in xsf]))

            axg.set_xlim(axf.get_xlim())
            xsg = np.linspace(axg.get_xlim()[0], axg.get_xlim()[1], 900)
            g_curve.set_data(xsg, np.array([g_vis(float(x)) for x in xsg]))

            prev = steps[i-1] if i > 0 else st

            for j in range(frames_per_step):
                t = j/(frames_per_step-1) if frames_per_step > 1 else 1.0
                tt = _ease(t)

                a_vis = prev.a + (st.a - prev.a) * tt
                b_vis = prev.b + (st.b - prev.b) * tt

                v_a.set_xdata([a_vis, a_vis])
                v_b.set_xdata([b_vis, b_vis])

                ga_vis = g_vis(a_vis)
                gb_vis = g_vis(b_vis)
                chord_line.set_data([a_vis, b_vis], [ga_vis, gb_vis])

                pt.set_data([st.x], [g_vis(st.x)])
                pt_f.set_data([st.x], [f_vis(st.x)])

                txt.set_text(
                    f"k={i+1}/{len(steps)}\n[a,b]=[{st.a:.5f},{st.b:.5f}]\n"
                    f"g(a)={g_vis(st.a):+.3f}, g(b)={g_vis(st.b):+.3f}\n"
                    f"x={st.x:.6f}, g(x)={g_vis(st.x):+.3e}"
                )

                writer.grab_frame()

            for _ in range(hold_frames):
                writer.grab_frame()

    plt.close(fig)


def render_newton_video(steps: List[NewtonStep], out_file: Path, fps: int, dpi: int,
                        frames_per_step: int = 40, hold_frames: int = 18):
    if not steps:
        return
    frames_per_step, hold_frames = _adaptive_frames(len(steps), fps, frames_per_step, hold_frames)
    xs_all = [st.x for st in steps] + [steps[-1].x_next]
    L = min(xs_all); R = max(xs_all)
    xlim0 = (L - 0.8, R + 0.8)

    fig, (axf, axg) = plt.subplots(2, 1, figsize=(7.2, 6.8), constrained_layout=True)
    setup_ax_1d(axf, "4) Ньютон по g(x)=f'(x) — сверху f(x)")
    f_line = draw_function_1d(axf, xlim0)

    axg.set_facecolor(BG)
    axg.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    axg.set_xlabel("x")
    axg.set_ylabel("g(x)=f'(x)")
    axg.set_title("Касательная к g(x) в x_k", color=DARK, fontsize=11, fontweight="bold")
    axg.axhline(0, color=DARK, linewidth=1.0, alpha=0.45)

    xs = np.linspace(xlim0[0], xlim0[1], 700)
    gs = np.array([df1d(float(x)) for x in xs])
    axg.plot(xs, gs, color=ITMO_BLUE, linewidth=2.2, alpha=0.9)

    pt, = axg.plot([], [], "o", color=ITMO_PINK, markersize=6)
    pt_next, = axg.plot([], [], "o", color=ITMO_LIME, markersize=6)
    tan_line, = axg.plot([], [], color=ITMO_ORANGE, linewidth=2.0, alpha=0.95)
    pt_f, = axf.plot([], [], "o", color=ITMO_PINK, markersize=6)

    txt = axg.text(0.02, 0.98, "", transform=axg.transAxes, va="top", ha="left",
                   fontsize=10, color=DARK,
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(fps=fps, metadata={"title": out_file.stem}, bitrate=1800, extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"])
    with writer.saving(fig, str(out_file), dpi=dpi):
        for i, st in enumerate(steps):
            _progress("newton", i, len(steps))
            # зум вокруг текущего шага (чтобы движение видно)
            zL = min(st.x, st.x_next) - 0.8
            zR = max(st.x, st.x_next) + 0.8
            axf.set_xlim(zL, zR)
            zoom_ylim_from_xlim(axf)
            axg.set_xlim(axf.get_xlim())

            dx = (axg.get_xlim()[1] - axg.get_xlim()[0]) * 0.45
            xA, xB = st.x - dx, st.x + dx
            yA = st.gx + st.gpx * (xA - st.x)
            yB = st.gx + st.gpx * (xB - st.x)

            for j in range(frames_per_step):
                t = j/(frames_per_step-1) if frames_per_step > 1 else 1.0
                tt = _ease(t)
                x_vis = st.x + (st.x_next - st.x) * tt
                gx_vis = df1d(x_vis)
                pt.set_data([x_vis], [gx_vis])
                pt_f.set_data([x_vis], [f1d(x_vis)])
                tan_line.set_data([xA, xB], [yA, yB])
                pt_next.set_data([st.x_next], [df1d(st.x_next)])
                txt.set_text(
                    f"step {i+1}/{len(steps)}\n"
                    f"xk={st.x:.5f}\n"
                    f"g(xk)={st.gx:.5f}\n"
                    f"g'(xk)={st.gpx:.5f}\n"
                    f"x(k+1)={st.x_next:.5f}"
                )
                writer.grab_frame()
            for _ in range(hold_frames):
                writer.grab_frame()
    plt.close(fig)

def render_quad_approx_video(steps: List[QuadApproxStep], out_file: Path, fps: int, dpi: int,
                             frames_per_step: int = 18, hold_frames: int = 10):
    if not steps:
        return
    f_vis = f1d_quad_demo if USE_DEMO_QUAD_FUNC else f1d
    frames_per_step, hold_frames = _adaptive_frames(len(steps), fps, frames_per_step, hold_frames)
    st0 = steps[0]
    xL0 = min(st0.x1, st0.x3)
    xR0 = max(st0.x1, st0.x3)
    xlim0 = (xL0 - 0.8, xR0 + 0.8)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    setup_ax_1d(ax, "5) Квадратичная аппроксимация")
    f_line = draw_function_1d(ax, xlim0)
    xs0 = np.linspace(xlim0[0], xlim0[1], 900)
    f_line.set_data(xs0, np.array([f_vis(float(x)) for x in xs0]))

    p1, = ax.plot([], [], "o", color=ITMO_PINK, markersize=6)
    p2, = ax.plot([], [], "o", color=ITMO_PINK, markersize=6)
    p3, = ax.plot([], [], "o", color=ITMO_PINK, markersize=6)
    pv, = ax.plot([], [], "o", color=ITMO_LIME, markersize=7)
    parab, = ax.plot([], [], color=ITMO_ORANGE, linewidth=2.2, alpha=0.92)

    # span = Rectangle: обновлять через set_xy((x,y)) + set_width()
    span = ax.axvspan(xL0, xR0, color=ITMO_CYAN, alpha=0.12)

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    writer = FFMpegWriter(fps=fps, metadata={"title": out_file.stem}, bitrate=1800, extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"])
    with writer.saving(fig, str(out_file), dpi=dpi):
        for i, st in enumerate(steps):
            _progress("quad", i, len(steps))
            xL = min(st.x1, st.x3)
            xR = max(st.x1, st.x3)
            zoom_xlim(ax, xL, xR, pad_frac=0.55, min_width=0.04)
            zoom_ylim_from_xlim(ax)
            xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 900)
            f_line.set_data(xs, np.array([f_vis(float(x)) for x in xs]))

            span.set_xy((xL, 0))
            span.set_width(max(1e-12, xR - xL))

            for j in range(frames_per_step):
                t = j/(frames_per_step-1) if frames_per_step > 1 else 1.0
                tt = _ease(t)

                prev = steps[i-1] if i > 0 else st
                xbar_vis = prev.xbar + (st.xbar - prev.xbar) * tt
                fbar_vis = f_vis(xbar_vis)

                p1.set_data([st.x1], [st.f1])
                p2.set_data([st.x2], [st.f2])
                p3.set_data([st.x3], [st.f3])
                pv.set_data([xbar_vis], [fbar_vis])

                xs = np.array([st.x1, st.x2, st.x3], dtype=float)
                ys = np.array([st.f1, st.f2, st.f3], dtype=float)
                coef = np.polyfit(xs, ys, deg=2)
                x_plot = np.linspace(xL, xR, 240)
                y_plot = coef[0]*x_plot**2 + coef[1]*x_plot + coef[2]
                parab.set_data(x_plot, y_plot)

                txt.set_text(
                    f"k={i+1}/{len(steps)}\n"
                    f"x1={st.x1:.5f}, x2={st.x2:.5f}, x3={st.x3:.5f}\n"
                    f"x̄={st.xbar:.5f} ({'inside' if st.inside else 'outside'})\n"
                    f"best x={st.xmin:.5f}, f={st.fmin:.5f}"
                )
                writer.grab_frame()
            for _ in range(hold_frames):
                writer.grab_frame()
    plt.close(fig)


# ============================================================
# 9) Рендер 2D (уровни + траектория + нормаль/касательная)
# ============================================================
def render_2d_path_video(path: Path2D, title: str, out_file: Path, fps: int, dpi: int,
                         frames_per_step: int = 12, hold_frames: int = 6,
                         zoom: bool = False, zoom_recent: int = 6):
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

    levels = np.linspace(float(np.min(Z)), float(np.percentile(Z, 92)), 18)
    ax.contour(X, Y, Z, levels=levels, colors=ITMO_CYAN, linewidths=0.9, alpha=0.75)
    ax.contour(X, Y, Z, levels=7, colors=ITMO_BLUE, linewidths=1.1, alpha=0.35)

    line, = ax.plot([], [], color=ITMO_BLUE, linewidth=2.8, alpha=0.95)
    pt, = ax.plot([], [], "o", color=ITMO_PINK, markersize=7)
    normal = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1,
                       color=ITMO_ORANGE, width=0.013, zorder=6)
    tang = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1,
                     color=ITMO_LIME, width=0.013, zorder=6)

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, color=DARK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.95))

    def set_single_quiver(q, x, y, u, v):
        q.set_offsets(np.array([[x, y]]))
        q.set_UVC(np.array([u]), np.array([v]))

    writer = FFMpegWriter(fps=fps, metadata={"title": out_file.stem}, bitrate=1800, extra_args=["-preset","ultrafast","-crf","23","-pix_fmt","yuv420p"])
    with writer.saving(fig, str(out_file), dpi=dpi):
        for i in range(len(path.points)):
            # Прогресс по шагам (как в 1D)
            if i > 0:
                _progress(out_file.stem, i-1, len(path.points)-1)

            p_prev = path.points[i-1] if i > 0 else path.points[i]
            p_cur  = path.points[i]
            g_cur  = path.grads[i]

            # "Приближение" (зум) для 2D: делаем окно вокруг последних точек, чтобы было видно,
            # как метод подходит к минимуму.
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
                tt = _ease(t)
                x = p_prev[0] + (p_cur[0] - p_prev[0]) * tt
                y = p_prev[1] + (p_cur[1] - p_prev[1]) * tt

                pts = np.array(path.points[:i+1])
                line.set_data(pts[:, 0], pts[:, 1])
                pt.set_data([x], [y])

                # "Приближение" для покоординатного (и вообще 2D):
                # фокусируемся на последних нескольких точках, чтобы было видно, что метод доезжает.
                if zoom and i >= 1:
                    start = max(0, i - zoom_recent)
                    recent = np.array(path.points[start:i+1])
                    rxmin, rymin = recent.min(axis=0)
                    rxmax, rymax = recent.max(axis=0)
                    # добавим текущую интерполированную точку тоже
                    rxmin = min(rxmin, x); rxmax = max(rxmax, x)
                    rymin = min(rymin, y); rymax = max(rymax, y)
                    w = max(1e-6, rxmax - rxmin)
                    h = max(1e-6, rymax - rymin)
                    pad_x = max(0.35*w, 0.35)
                    pad_y = max(0.35*h, 0.35)
                    ax.set_xlim(rxmin - pad_x, rxmax + pad_x)
                    ax.set_ylim(rymin - pad_y, rymax + pad_y)

                ng = norm2(g_cur)
                if ng < 1e-12:
                    u1=v1=u2=v2=0.0
                else:
                    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
                    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
                    scale = 0.28 * float(min(xspan, yspan))
                    u1, v1 = (-g_cur[0]/ng)*scale, (-g_cur[1]/ng)*scale  # нормаль (направление спуска)
                    u2, v2 = (-v1, u1)                                   # касательная

                set_single_quiver(normal, x, y, u1, v1)
                set_single_quiver(tang, x, y, u2, v2)

                txt.set_text(
                    f"шаг {i}/{len(path.points)-1}\n"
                    f"f={f2d(x,y):.5f}\n"
                    f"||grad||={ng:.5f}\n"
                    f"{path.info[i] if i < len(path.info) else ''}"
                )
                writer.grab_frame()

            for _ in range(hold_frames):
                writer.grab_frame()
    plt.close(fig)


# ============================================================
# 10) Склейка mp4 последовательно (1→2→...→8)
# ============================================================
def concat_videos_ffmpeg(video_files: List[Path], out_file: Path):
    tmp_list = out_file.parent / "concat_list.txt"
    with tmp_list.open("w", encoding="utf-8") as f:
        for p in video_files:
            f.write(f"file '{p.resolve().as_posix()}'\n")

    cmd = [
        ffmpeg_cmd(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(tmp_list),
        "-c", "copy",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)


# ============================================================
# 11) CONFIG
# ============================================================
if USE_DEMO_1D_FOR_VIDEO:
    CONFIG_A1D = 0.0
    CONFIG_B1D = 4.0
    CONFIG_X0_NEWTON = 3.6
    CONFIG_X1_QUAD = 0.6
    CONFIG_H_QUAD = 0.7
    CONFIG_EPS_1D = 1e-3
else:
    CONFIG_A1D = 1.5
    CONFIG_B1D = 2.0
    CONFIG_X0_NEWTON = (CONFIG_A1D + CONFIG_B1D) / 2
    CONFIG_X1_QUAD = 1.5
    CONFIG_H_QUAD = 0.1
    CONFIG_EPS_1D = 2e-3

CONFIG_X0_2D = (2.0, -2.0)
CONFIG_EPS_2D = 1e-4
CONFIG_H0_GRAD = 0.25

CONFIG_FPS = 24
CONFIG_DPI = 140


# ============================================================
# 12) main
# ============================================================
def main():
    ensure_ffmpeg()
    out_dir = Path("videos_itmo_v5")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1D steps
    dich_steps = dichotomy_min(CONFIG_A1D, CONFIG_B1D, eps=CONFIG_EPS_1D)
    gold_steps = golden_min(CONFIG_A1D, CONFIG_B1D, eps=CONFIG_EPS_1D)
    chord_steps = chords_on_derivative(CONFIG_A1D, CONFIG_B1D, eps=CONFIG_EPS_1D)
    newt_steps = newton_on_derivative(CONFIG_X0_NEWTON, eps=CONFIG_EPS_1D, min_steps=6)
    quad_steps = quad_approx_min(CONFIG_A1D, CONFIG_B1D, x1=CONFIG_X1_QUAD, h=CONFIG_H_QUAD,
                                 eps=CONFIG_EPS_1D, min_steps=10, func=(f1d_quad_demo if USE_DEMO_QUAD_FUNC else f1d))

    # 2D paths
    # Для покоординатного: останавливаемся по ||x_n-x_{n-1}||<eps без искусственного "дотягивания"
    coord_path = coord_descent_2d(CONFIG_X0_2D, eps=CONFIG_EPS_2D, min_cycles_for_video=0)
    grad_path  = gradient_descent_2d(CONFIG_X0_2D, eps=CONFIG_EPS_2D, h0=CONFIG_H0_GRAD)
    steep_path = steepest_descent_2d(CONFIG_X0_2D, eps=CONFIG_EPS_2D)

    # render videos
    print("\n=== РЕНДЕР 1/8: дихотомия ===", flush=True)
    render_dichotomy_video(dich_steps, out_dir / "01_dichotomy.mp4", CONFIG_FPS, CONFIG_DPI)
    print("\n=== РЕНДЕР 2/8: золотое сечение ===", flush=True)
    render_golden_video(gold_steps, out_dir / "02_golden_section.mp4", CONFIG_FPS, CONFIG_DPI)
    print("\n=== РЕНДЕР 3/8: хорд по производной ===", flush=True)
    render_chords_video(chord_steps, out_dir / "03_chords.mp4", CONFIG_FPS, CONFIG_DPI)
    print("\n=== РЕНДЕР 4/8: Ньютон ===", flush=True)
    render_newton_video(newt_steps, out_dir / "04_newton.mp4", CONFIG_FPS, CONFIG_DPI)
    print("\n=== РЕНДЕР 5/8: квадратичная аппроксимация ===", flush=True)
    render_quad_approx_video(quad_steps, out_dir / "05_quadratic_approx.mp4", CONFIG_FPS, CONFIG_DPI)

    print("\n=== РЕНДЕР 6/8: покоординатный спуск ===", flush=True)
    render_2d_path_video(coord_path, "6) Покоординатный спуск: уровни + нормаль/касательная",
                         out_dir / "06_coord_descent.mp4", CONFIG_FPS, CONFIG_DPI,
                         zoom=True)
    print("\n=== РЕНДЕР 7/8: градиентный спуск ===", flush=True)
    render_2d_path_video(grad_path, "7) Градиентный спуск: уровни + нормаль/касательная",
                         out_dir / "07_gradient_descent.mp4", CONFIG_FPS, CONFIG_DPI)
    print("\n=== РЕНДЕР 8/8: наискорейший спуск ===", flush=True)
    render_2d_path_video(steep_path, "8) Наискорейший спуск: уровни + нормаль/касательная",
                         out_dir / "08_steepest_descent.mp4", CONFIG_FPS, CONFIG_DPI)

    parts = [
        out_dir / "01_dichotomy.mp4",
        out_dir / "02_golden_section.mp4",
        out_dir / "03_chords.mp4",
        out_dir / "04_newton.mp4",
        out_dir / "05_quadratic_approx.mp4",
        out_dir / "06_coord_descent.mp4",
        out_dir / "07_gradient_descent.mp4",
        out_dir / "08_steepest_descent.mp4",
    ]
    concat_videos_ffmpeg(parts, out_dir / "combined_sequential.mp4")

    print("\nГотово:", out_dir.resolve())
    print("1D режим:", "DEMO" if USE_DEMO_1D_FOR_VIDEO else "LAB")
    print("FFmpeg:", ffmpeg_cmd())

if __name__ == "__main__":
    main()
