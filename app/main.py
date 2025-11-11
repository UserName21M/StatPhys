import os
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from simulation.model import SimulationModel
from ui.pages import TitlePage, TheoryPage, AuthorsPage, PresentationPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Презентация: Столкновения частиц в коробке")
        self.resize(1280, 800)

        self.model = SimulationModel()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.title = TitlePage(self.stack)
        self.theory = TheoryPage(self.stack)
        self.authors = AuthorsPage(self.stack)
        self.presentation = PresentationPage(self.model, self.stack)

        self.stack.addWidget(self.title)
        self.stack.addWidget(self.presentation)
        self.stack.addWidget(self.theory)
        self.stack.addWidget(self.authors)

        self.title.btn_presentation.clicked.connect(lambda: self.stack.setCurrentWidget(self.presentation))
        self.title.btn_theory.clicked.connect(lambda: self.stack.setCurrentWidget(self.theory))
        self.title.btn_authors.clicked.connect(lambda: self.stack.setCurrentWidget(self.authors))
        self.title.btn_exit.clicked.connect(self.close)

        self.stack.setCurrentWidget(self.title)


def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

