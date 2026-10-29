"""A small generated picture for video calls: name, state and a level meter, rendered with Pillow."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw

from .audio import rgb_to_i420


class Avatar:
    def __init__(self, width: int = 640, height: int = 360, name: str = "Hermes"):
        self.w, self.h, self.name = width - width % 2, height - height % 2, name
        self.state = "listening"
        self.level = 0.0
        self.caption = ""

    def frame_rgb(self) -> np.ndarray:
        img = Image.new("RGB", (self.w, self.h), (18, 20, 28))
        d = ImageDraw.Draw(img)
        cx, cy = self.w // 2, self.h // 2 - 20
        pulse = 1.0 + 0.25 * min(1.0, self.level * 8) * (0.5 + 0.5 * np.sin(time.time() * 6))
        r = int(min(self.w, self.h) * 0.18 * pulse)
        color = {"listening": (70, 160, 255), "thinking": (255, 190, 60), "speaking": (90, 220, 140)}.get(self.state, (160, 160, 160))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=6)
        d.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), fill=color)
        d.text((cx - 4 * len(self.name), cy + r + 16), self.name, fill=(230, 230, 230))
        d.text((cx - 4 * len(self.state), cy + r + 34), self.state, fill=color)
        if self.caption:
            d.text((16, self.h - 28), self.caption[:90], fill=(200, 200, 200))
        bars = int(min(1.0, self.level * 10) * 20)
        for i in range(20):
            x0 = 16 + i * 12
            d.rectangle((x0, 16, x0 + 8, 28), fill=color if i < bars else (50, 52, 60))
        return np.asarray(img, dtype=np.uint8)

    def frame_i420(self) -> bytes:
        return rgb_to_i420(self.frame_rgb())
