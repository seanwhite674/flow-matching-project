"""Minimal helpers for writing .excalidraw files from Python."""

import json
import random
import time


class Drawing:
    def __init__(self, seed=0):
        self.elements, self.files = [], {}
        self.rng = random.Random(seed)
        self.now = int(time.time() * 1000)

    def _base(self, kind, x, y, w, h, **kw):
        el = {
            "id": f"{kind}-{len(self.elements)}", "type": kind, "x": x, "y": y,
            "width": w, "height": h, "angle": 0, "strokeColor": "#1e1e1e",
            "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
            "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
            "frameId": None, "roundness": None, "seed": self.rng.randint(1, 2**31),
            "version": 1, "versionNonce": self.rng.randint(1, 2**31), "isDeleted": False,
            "boundElements": None, "updated": self.now, "link": None, "locked": False,
        }
        el.update(kw)
        self.elements.append(el)
        return el

    def text(self, x, y, s, size=20, align="center", color="#1e1e1e"):
        w = max(len(s) * size * 0.55, 10)
        return self._base("text", x - (w / 2 if align == "center" else 0), y, w, size * 1.25,
                          text=s, originalText=s, fontSize=size, fontFamily=1,
                          textAlign=align, verticalAlign="top", containerId=None,
                          lineHeight=1.25, autoResize=True, strokeColor=color)

    def polyline(self, pts, color="#1e1e1e", width=2, arrow=False, dashed=False, roughness=1):
        """Open line through absolute points [(x, y), ...]; optional arrowhead at the end."""
        x0, y0 = pts[0]
        rel = [[px - x0, py - y0] for px, py in pts]
        xs, ys = [p[0] for p in rel], [p[1] for p in rel]
        return self._base("arrow" if arrow else "line", x0, y0,
                          max(xs) - min(xs), max(ys) - min(ys),
                          points=rel, lastCommittedPoint=None, startBinding=None,
                          endBinding=None, startArrowhead=None,
                          endArrowhead="arrow" if arrow else None, elbowed=False,
                          strokeColor=color, strokeWidth=width, roughness=roughness,
                          strokeStyle="dashed" if dashed else "solid",
                          roundness={"type": 2})

    def dot(self, x, y, r=6, color="#1e1e1e"):
        return self._base("ellipse", x - r, y - r, 2 * r, 2 * r, strokeColor=color,
                          backgroundColor=color, fillStyle="hachure", roughness=1)

    def save(self, path):
        doc = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
               "elements": self.elements,
               "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
               "files": self.files}
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
