# theory/render.py
import io
from typing import Tuple
from PyQt6.QtGui import QPixmap
from matplotlib.figure import Figure

def latex_to_pixmap(tex: str, dpi: int = 160, pad_inches: float = 0.15) -> QPixmap:
    """
    Рендер LaTeX-формулы в QPixmap с использованием matplotlib.mathtext.
    Без прямого доступа к renderer; обрезка через bbox_inches='tight'.
    """
    fig = Figure(figsize=(0.01, 0.01), dpi=dpi)
    # Создаём ось во весь холст
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    # Центрируем формулу
    ax.text(0.5, 0.5, f"${tex}$", ha="center", va="center")
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    buf.seek(0)
    pm = QPixmap()
    pm.loadFromData(buf.read())
    return pm

def title_with_caption(title: str, caption: str) -> Tuple[QPixmap, QPixmap]:
    """
    Вспомогательный: заголовок формулой и подпись текстом.
    """
    title_pm = latex_to_pixmap(title, dpi=180)
    caption_pm = latex_to_pixmap(r"\text{" + caption + r"}", dpi=120)
    return title_pm, caption_pm
