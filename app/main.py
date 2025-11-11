# app/main.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from simulation.model import SimulationModel
from ui.pages import TitlePage, TheoryPage, AuthorsPage, PresentationPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Презентация: Столкновения частиц в коробке")
        self.resize(1280, 800)

        # 1) Модель
        self.model = SimulationModel()

        # 2) Стек страниц
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 3) Страницы (ВАЖНО: создать до меню)
        self.title = TitlePage(self.stack)
        self.presentation = PresentationPage(self.model, self.stack)
        self.theory = TheoryPage(self.stack)
        self.authors = AuthorsPage(self.stack)

        # 4) Добавить в стек
        self.stack.addWidget(self.title)
        self.stack.addWidget(self.presentation)
        self.stack.addWidget(self.theory)
        self.stack.addWidget(self.authors)

        # 5) Навигация с титула
        self.title.btn_presentation.clicked.connect(lambda: self.stack.setCurrentWidget(self.presentation))
        self.title.btn_theory.clicked.connect(lambda: self.stack.setCurrentWidget(self.theory))
        self.title.btn_authors.clicked.connect(lambda: self.stack.setCurrentWidget(self.authors))
        self.title.btn_exit.clicked.connect(self.close)

        # 6) Стартовая страница
        self.stack.setCurrentWidget(self.title)

        # 7) Меню (после создания страниц)
        self._create_menu()

    def _create_menu(self):
        menubar = self.menuBar()

        m_app = menubar.addMenu("Приложение")
        act_exit = m_app.addAction("Выход")
        act_exit.triggered.connect(self.close)

        m_view = menubar.addMenu("Страницы")
        act_title = m_view.addAction("Титул")
        act_title.triggered.connect(lambda: self.stack.setCurrentWidget(self.title))
        act_pres = m_view.addAction("Презентация")
        act_pres.triggered.connect(lambda: self.stack.setCurrentWidget(self.presentation))
        act_theor = m_view.addAction("Теория")
        act_theor.triggered.connect(lambda: self.stack.setCurrentWidget(self.theory))
        act_auth = m_view.addAction("Авторы")
        act_auth.triggered.connect(lambda: self.stack.setCurrentWidget(self.authors))

        m_sim = menubar.addMenu("Симуляция")
        act_start = m_sim.addAction("Старт")
        act_start.triggered.connect(self.presentation.start)
        act_pause = m_sim.addAction("Пауза")
        act_pause.triggered.connect(self.presentation.pause)
        act_reset = m_sim.addAction("Сброс")
        act_reset.triggered.connect(self.presentation.reset)

        m_help = menubar.addMenu("Справка")
        act_about = m_help.addAction("О программе")
        def _about():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "О программе",
                "Столкновения частиц в коробке\nБроуновское движение\nPyQt6 + Matplotlib"
            )
        act_about.triggered.connect(_about)

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
