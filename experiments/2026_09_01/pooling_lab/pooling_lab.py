"""Interactive kernel + pooling playground.

Left panel  : the source image (cat.jpg, greyscale).
Middle panel: the image after one kernel is swiped over it (convolution).
Right panel : that result after pooling.

Everything recomputes as you move a control, so you can watch what pooling
does to a filtered image.

    uv run experiments/2026_09_01/pooling_lab/pooling_lab.py
    # or: .venv/bin/python experiments/2026_09_01/pooling_lab/pooling_lab.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageTk
from scipy import ndimage
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
IMAGE_PATH = HERE / "cat.jpg"
WORK_SIZE = 384          # image is resized so the long side is this many pixels
VIEW_SIZE = 340          # how big each panel is drawn


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

def _gauss1d(n: int) -> np.ndarray:
    x = np.arange(n) - (n - 1) / 2.0
    sigma = max(n / 6.0, 0.5)
    g = np.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def _dgauss1d(n: int) -> np.ndarray:
    x = np.arange(n) - (n - 1) / 2.0
    sigma = max(n / 6.0, 0.5)
    g = np.exp(-(x**2) / (2 * sigma**2))
    d = -x * g
    scale = np.abs(d).sum()
    return d / scale if scale else d


def make_kernel(name: str, n: int) -> np.ndarray:
    n = max(1, n | 1)                       # force odd
    mid = n // 2
    k = np.zeros((n, n), dtype=np.float64)

    if name == "horizontal line":           # a row of ones swiped across
        k[mid, :] = 1.0 / n
    elif name == "vertical line":
        k[:, mid] = 1.0 / n
    elif name == "diagonal line":
        idx = np.arange(n)
        k[idx, idx] = 1.0 / n
    elif name == "cross":
        k[mid, :] = 1.0
        k[:, mid] = 1.0
        k /= k.sum()
    elif name == "box blur":
        k[:] = 1.0 / (n * n)
    elif name == "gaussian blur":
        g = _gauss1d(n)
        k = np.outer(g, g)
    elif name == "edge horizontal":         # bright above, dark below
        k = np.outer(_dgauss1d(n), _gauss1d(n))
    elif name == "edge vertical":
        k = np.outer(_gauss1d(n), _dgauss1d(n))
    elif name == "laplacian":
        g = _gauss1d(n)
        blur = np.outer(g, g)
        k = -blur
        k[mid, mid] += 1.0
    elif name == "sharpen":
        g = _gauss1d(n)
        blur = np.outer(g, g)
        k = -blur
        k[mid, mid] += 2.0
    elif name == "emboss":
        k = np.outer(_dgauss1d(n), _gauss1d(n)) + np.outer(_gauss1d(n), _dgauss1d(n))
    elif name == "identity":
        k[mid, mid] = 1.0
    else:
        raise ValueError(name)
    return k


KERNELS = [
    "horizontal line", "vertical line", "diagonal line", "cross",
    "box blur", "gaussian blur", "edge horizontal", "edge vertical",
    "laplacian", "sharpen", "emboss", "identity",
]


# --------------------------------------------------------------------------
# maths
# --------------------------------------------------------------------------

def swipe(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Slide the kernel over every position of the image."""
    if k.shape[0] >= 13:                                  # big kernels: go via FFT
        return fftconvolve(img, k[::-1, ::-1], mode="same")
    return ndimage.convolve(img, k, mode="reflect")


def rectify(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "abs":
        return np.abs(img)
    if mode == "relu":
        return np.maximum(img, 0.0)
    return img


def pool(img: np.ndarray, size: int, stride: int, mode: str) -> np.ndarray:
    size = max(1, size)
    stride = max(1, stride)
    if size == 1 and stride == 1:
        return img
    h, w = img.shape
    if size > h or size > w:
        size = min(h, w)
    win = sliding_window_view(img, (size, size))[::stride, ::stride]
    if mode == "max":
        return win.max(axis=(-1, -2))
    if mode == "min":
        return win.min(axis=(-1, -2))
    return win.mean(axis=(-1, -2))


def to_photo(img: np.ndarray, normalise: bool, upscale_to: int | None) -> ImageTk.PhotoImage:
    a = img.astype(np.float64)
    if normalise:
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    else:
        a = np.clip(a, 0.0, 1.0)
    pic = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    if upscale_to:
        pic = pic.resize((upscale_to, upscale_to * pic.height // pic.width or 1),
                         Image.NEAREST)
    return ImageTk.PhotoImage(pic)


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk, source: np.ndarray) -> None:
        self.root = root
        self.source = source
        self.pending: str | None = None
        self.refs: list[ImageTk.PhotoImage] = []

        self.kernel_name = tk.StringVar(value="horizontal line")
        self.kernel_size = tk.IntVar(value=9)
        self.rectify_mode = tk.StringVar(value="abs")
        self.pool_mode = tk.StringVar(value="max")
        self.pool_size = tk.IntVar(value=4)
        self.pool_stride = tk.IntVar(value=4)
        self.link_stride = tk.BooleanVar(value=True)
        self.normalise = tk.BooleanVar(value=True)
        self.upscale = tk.BooleanVar(value=True)

        self._build()
        self.recompute()

    # -- layout ------------------------------------------------------------

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        side = ttk.Frame(outer)
        side.pack(side="left", fill="y", padx=(0, 10))

        # kernel controls
        box = ttk.LabelFrame(side, text="kernel", padding=8)
        box.pack(fill="x")
        ttk.Label(box, text="type").pack(anchor="w")
        combo = ttk.Combobox(box, values=KERNELS, textvariable=self.kernel_name,
                             state="readonly", width=18)
        combo.pack(fill="x", pady=(0, 6))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.schedule())

        self.kernel_size_label = ttk.Label(box, text="size 9 x 9")
        self.kernel_size_label.pack(anchor="w")
        ttk.Scale(box, from_=1, to=31, variable=self.kernel_size,
                  command=lambda _v: self.on_kernel_size()).pack(fill="x")

        self.kernel_canvas = tk.Canvas(box, width=96, height=96,
                                       highlightthickness=1, bg="#222")
        self.kernel_canvas.pack(pady=6)

        ttk.Label(box, text="rectify output").pack(anchor="w")
        for label, value in (("none (signed)", "none"), ("abs", "abs"), ("relu", "relu")):
            ttk.Radiobutton(box, text=label, value=value, variable=self.rectify_mode,
                            command=self.schedule).pack(anchor="w")

        # pooling controls
        box = ttk.LabelFrame(side, text="pooling", padding=8)
        box.pack(fill="x", pady=(10, 0))
        for label in ("max", "avg", "min"):
            ttk.Radiobutton(box, text=label, value=label, variable=self.pool_mode,
                            command=self.schedule).pack(anchor="w")

        self.pool_size_label = ttk.Label(box, text="window 4 x 4")
        self.pool_size_label.pack(anchor="w", pady=(6, 0))
        ttk.Scale(box, from_=1, to=32, variable=self.pool_size,
                  command=lambda _v: self.on_pool_size()).pack(fill="x")

        self.pool_stride_label = ttk.Label(box, text="stride 4")
        self.pool_stride_label.pack(anchor="w", pady=(6, 0))
        self.stride_scale = ttk.Scale(box, from_=1, to=32, variable=self.pool_stride,
                                      command=lambda _v: self.on_pool_stride())
        self.stride_scale.pack(fill="x")
        ttk.Checkbutton(box, text="stride follows window", variable=self.link_stride,
                        command=self.on_pool_size).pack(anchor="w")

        # display controls
        box = ttk.LabelFrame(side, text="display", padding=8)
        box.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(box, text="normalise contrast", variable=self.normalise,
                        command=self.schedule).pack(anchor="w")
        ttk.Checkbutton(box, text="blow pooled map back up", variable=self.upscale,
                        command=self.schedule).pack(anchor="w")

        # panels
        panels = ttk.Frame(outer)
        panels.pack(side="left", fill="both", expand=True)
        self.panels = {}
        for i, (key, title) in enumerate((("src", "original"),
                                          ("conv", "after kernel"),
                                          ("pool", "after pooling"))):
            frame = ttk.Frame(panels)
            frame.grid(row=0, column=i, padx=6)
            head = ttk.Label(frame, text=title, font=("TkDefaultFont", 10, "bold"))
            head.pack()
            canvas = tk.Label(frame, background="#111")
            canvas.pack()
            info = ttk.Label(frame, text="")
            info.pack()
            self.panels[key] = (canvas, info)

    # -- control callbacks -------------------------------------------------

    def on_kernel_size(self) -> None:
        n = max(1, self.kernel_size.get() | 1)
        self.kernel_size_label.config(text=f"size {n} x {n}")
        self.schedule()

    def on_pool_size(self) -> None:
        n = self.pool_size.get()
        self.pool_size_label.config(text=f"window {n} x {n}")
        if self.link_stride.get():
            self.pool_stride.set(n)
            self.stride_scale.state(["disabled"])
        else:
            self.stride_scale.state(["!disabled"])
        self.pool_stride_label.config(text=f"stride {self.pool_stride.get()}")
        self.schedule()

    def on_pool_stride(self) -> None:
        self.pool_stride_label.config(text=f"stride {self.pool_stride.get()}")
        self.schedule()

    def schedule(self) -> None:
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(60, self.recompute)

    # -- work --------------------------------------------------------------

    def recompute(self) -> None:
        self.pending = None
        self.refs.clear()

        k = make_kernel(self.kernel_name.get(), self.kernel_size.get())
        conv = rectify(swipe(self.source, k), self.rectify_mode.get())
        pooled = pool(conv, self.pool_size.get(), self.pool_stride.get(),
                      self.pool_mode.get())

        self.draw_kernel(k)
        self.show("src", self.source, VIEW_SIZE)
        self.show("conv", conv, VIEW_SIZE)
        self.show("pool", pooled, VIEW_SIZE if self.upscale.get() else None)

    def show(self, key: str, img: np.ndarray, upscale_to: int | None) -> None:
        photo = to_photo(img, self.normalise.get(), upscale_to)
        self.refs.append(photo)
        canvas, info = self.panels[key]
        canvas.config(image=photo)
        info.config(text=f"{img.shape[1]} x {img.shape[0]}   "
                         f"[{img.min():+.2f}, {img.max():+.2f}]")

    def draw_kernel(self, k: np.ndarray) -> None:
        self.kernel_canvas.delete("all")
        peak = float(np.abs(k).max()) or 1.0
        n = k.shape[0]
        cell = 96 / n
        for r in range(n):
            for c in range(n):
                v = k[r, c] / peak
                shade = int(abs(v) * 255)
                colour = f"#{shade:02x}{shade:02x}{shade:02x}" if v >= 0 \
                    else f"#{shade:02x}00{shade // 2:02x}"
                self.kernel_canvas.create_rectangle(
                    c * cell, r * cell, (c + 1) * cell, (r + 1) * cell,
                    fill=colour, outline="")


def load_image() -> np.ndarray:
    pic = Image.open(IMAGE_PATH).convert("L")
    scale = WORK_SIZE / max(pic.size)
    pic = pic.resize((max(1, int(pic.width * scale)), max(1, int(pic.height * scale))),
                     Image.LANCZOS)
    return np.asarray(pic, dtype=np.float64) / 255.0


def main() -> None:
    root = tk.Tk()
    root.title("kernel + pooling lab")
    App(root, load_image())
    root.mainloop()


if __name__ == "__main__":
    main()
