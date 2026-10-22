"""
Build panel 1 of the flow-matching intro as an Excalidraw file with real MNIST
digits embedded:

    p_data (unknown) --i.i.d.--> z_1 z_2 z_3 ... z_n
    x_0 ~ N(0, I)    -->  [ ??? ]  -->  new z ~ p_data

Run from the project root:
    .venv/bin/python slide_show/figures/make_mnist_excalidraw.py
Open the output in VS Code (Excalidraw extension) and export to PNG/SVG.
"""

import base64
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

OUT = Path(__file__).parent / "mnist_samples.excalidraw"
DATA_DIR = Path(__file__).parent / ".mnist_data"
IMG = 90          # displayed image size (px)
NOW = int(time.time() * 1000)
rng = random.Random(0)

elements, files = [], {}


def base(kind, x, y, w, h, **kw):
    el = {
        "id": f"{kind}-{len(elements)}", "type": kind, "x": x, "y": y,
        "width": w, "height": h, "angle": 0, "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
        "frameId": None, "roundness": None, "seed": rng.randint(1, 2**31),
        "version": 1, "versionNonce": rng.randint(1, 2**31), "isDeleted": False,
        "boundElements": None, "updated": NOW, "link": None, "locked": False,
    }
    el.update(kw)
    elements.append(el)
    return el


def text(x, y, s, size=20, align="center", width=None, color="#1e1e1e"):
    w = width or max(len(s) * size * 0.55, 10)
    return base("text", x - (w / 2 if align == "center" else 0), y, w, size * 1.25,
                text=s, originalText=s, fontSize=size, fontFamily=1,
                textAlign=align, verticalAlign="top", containerId=None,
                lineHeight=1.25, autoResize=True, strokeColor=color)


def arrow(x1, y1, x2, y2, dashed=False):
    return base("arrow", x1, y1, x2 - x1, y2 - y1,
                points=[[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint=None,
                startBinding=None, endBinding=None, startArrowhead=None,
                endArrowhead="arrow", elbowed=False,
                strokeStyle="dashed" if dashed else "solid",
                roundness={"type": 2})


def image(x, y, pixels, opacity=100):
    """Embed a 28x28 array in [0, 255] as a crisp (nearest-neighbour) PNG."""
    png = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).resize((112, 112), Image.NEAREST).save(png, "PNG")
    file_id = f"file-{len(files)}"
    files[file_id] = {
        "mimeType": "image/png", "id": file_id, "created": NOW,
        "dataURL": "data:image/png;base64," + base64.b64encode(png.getvalue()).decode(),
    }
    return base("image", x, y, IMG, IMG, fileId=file_id, status="saved",
                scale=[1, 1], strokeColor="transparent", opacity=opacity)


# ------------------------------------------------------------------ data
mnist = MNIST(DATA_DIR, train=True, download=True)
digits = [np.array(mnist[i][0]) for i in (0, 1, 2, 3)]   # a 5, 0, 4, 1
new_digit = np.array(mnist[7][0])                         # shown faded as the goal
noise = np.clip(np.random.default_rng(0).normal(0.5, 0.25, (28, 28)), 0, 1) * 255

# ------------------------------------------------------------------ layout
text(450, -70, "Generative modelling: learn to sample from p_data", size=28)

# row 1: unknown distribution and its samples
row1 = 40
base("ellipse", 0, row1 - 5, 200, 100, strokeStyle="dashed", backgroundColor="#e7f5ff",
     roundness={"type": 2})
text(100, row1 + 22, "p_data", size=26)
text(100, row1 + 55, "(unknown)", size=16, color="#868e96")
arrow(215, row1 + 45, 335, row1 + 45)
text(275, row1 + 10, "i.i.d.", size=16, color="#868e96")

xs = [355, 465, 575, 685]
for x, d in zip(xs[:3], digits[:3]):
    image(x, row1, d)
text(xs[3] + IMG / 2, row1 + 25, "…", size=36)
image(795, row1, digits[3])
for x, label in zip(xs[:3] + [795], ["z₁", "z₂", "z₃", "zₙ"]):
    text(x + IMG / 2, row1 + IMG + 8, label, size=22)

# row 2: the question
row2 = 260
image(30, row2, noise)
text(30 + IMG / 2, row2 + IMG + 8, "x₀ ~ N(0, I)", size=20)
arrow(140, row2 + 45, 300, row2 + 45)
base("rectangle", 315, row2 - 5, 220, 100, strokeStyle="dashed", backgroundColor="#fff3bf",
     roundness={"type": 3})
text(425, row2 + 20, "???", size=44)
arrow(550, row2 + 45, 700, row2 + 45)
image(715, row2, new_digit, opacity=45)
text(715 + IMG / 2, row2 + IMG + 8, "new z ~ p_data", size=20)

doc = {
    "type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
    "files": files,
}
OUT.write_text(json.dumps(doc, indent=1))
print(f"Wrote {OUT} ({len(elements)} elements, {len(files)} images)")
