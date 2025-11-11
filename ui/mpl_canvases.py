from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Mpl3DCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(5, 5), tight_layout=True)
        self.ax = fig.add_subplot(111, projection="3d")
        super().__init__(fig)


class Mpl2DCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(5, 5), tight_layout=True)
        self.ax1 = fig.add_subplot(211)  # Энергия
        self.ax2 = fig.add_subplot(212)  # Температура и скорость
        super().__init__(fig)
