# main.py
# Презентация "Столкновения частиц в коробке" (3D + графики) с интерактивом и титульной страницей.
# Зависимости: Python 3.10+, numpy, matplotlib, PyQt6
# Условные (приведённые) единицы: длина (m_r), масса (kg_r), время (s_r), энергия (E_r), температура (T_r), где k_B = 1.

import os
import sys
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QSpinBox, QDoubleSpinBox, QStackedWidget,
    QGroupBox, QFormLayout, QCheckBox, QSplitter, QScrollArea, QFileDialog
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (нужно для 3D)

# --------------------------
# Модель симуляции
# --------------------------

class SimulationModel:
    SPHERE, CYLINDER, PLANE = 0, 1, 2

    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)

        # Границы коробки (полудиагонали) в приведённых единицах
        self.box_half = np.array([5.0, 5.0, 5.0], dtype=float)

        # Параметры по умолчанию
        self.N = 100
        self.small_radius = 0.1
        self.small_mass = 1.0

        self.big_type = self.SPHERE
        self.big_radius = 1.0
        self.big_mass = 10.0
        self.big_resolution = 12

        self.dt = 0.01  # s_r
        self.init_speed = 20.0  # m_r/s_r

        # Состояние
        self.big_pos = np.zeros(3, dtype=float)
        self.big_vel = np.zeros(3, dtype=float)
        self.small_pos = None
        self.small_vel = None

        # Предрасчёт сетки большой фигуры
        self.big_mask = np.array([1, 1, 1], dtype=float)
        self.big_x = None
        self.big_y = None
        self.big_z = None

        # Инициализация
        self._init_particles()
        self._update_big_geometry()

        # Счётчики/истории
        self.time = 0.0
        self.collisions = 0

    def _init_particles(self):
        # Равномерно в коробке с учётом радиуса
        self.small_pos = (self.rng.random((self.N, 3)) - 0.5) * (2 * self.box_half - 2 * self.small_radius)
        # Случайные направления, нормированные, с масштабом скорости
        v = self.rng.normal(size=(self.N, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        v *= self.init_speed
        self.small_vel = v

        self.big_pos[:] = 0.0
        self.big_vel[:] = 0.0
        self.time = 0.0
        self.collisions = 0

    def set_params(self, N=None, dt=None, init_speed=None, small_radius=None, small_mass=None,
                   big_type=None, big_radius=None, big_mass=None, big_resolution=None):
        # Проверки диапазонов, чтобы избежать нечисленных режимов
        if N is not None:
            self.N = int(np.clip(N, 1, 20000))
        if dt is not None:
            self.dt = float(np.clip(dt, 1e-5, 0.1))
        if init_speed is not None:
            self.init_speed = float(np.clip(init_speed, 0.0, 1000.0))
        if small_radius is not None:
            self.small_radius = float(np.clip(small_radius, 1e-3, min(self.box_half) * 0.5))
        if small_mass is not None:
            self.small_mass = float(np.clip(small_mass, 1e-6, 1e6))
        if big_type is not None:
            self.big_type = int(big_type)
        if big_radius is not None:
            self.big_radius = float(np.clip(big_radius, 1e-3, min(self.box_half)))
        if big_mass is not None:
            self.big_mass = float(np.clip(big_mass, 1e-6, 1e9))
        if big_resolution is not None:
            self.big_resolution = int(np.clip(big_resolution, 6, 128))

        self._update_big_geometry()

    def reset(self):
        self._init_particles()

    # Геометрия большой фигуры
    def _create_sphere(self):
        phi = np.linspace(0, 2 * np.pi, self.big_resolution)
        theta = np.linspace(np.pi, 0, max(self.big_resolution // 2, 3))
        phi, theta = np.meshgrid(phi, theta)
        x = self.big_radius * np.cos(phi) * np.sin(theta)
        y = self.big_radius * np.sin(phi) * np.sin(theta)
        z = self.big_radius * np.cos(theta)
        return x, y, z

    def _create_cylinder(self):
        theta = np.linspace(0, 2 * np.pi, self.big_resolution)
        z = np.linspace(-self.box_half[-1], self.box_half[-1], 2)
        theta, z = np.meshgrid(theta, z)
        x = self.big_radius * np.cos(theta)
        y = self.big_radius * np.sin(theta)
        return x, y, z

    def _create_plane(self):
        # Прямоугольник в YZ с шириной 2*box_half
        x = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=float) * self.big_radius * 2 - self.big_radius
        y = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]], dtype=float) * self.box_half[1] * 2 - self.box_half[1]
        z = np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=float) * self.box_half[2] * 2 - self.box_half[2]
        return x, y, z

    def _update_big_geometry(self):
        if self.big_type == self.SPHERE:
            self.big_x, self.big_y, self.big_z = self._create_sphere()
            self.big_mask = np.array([1, 1, 1], dtype=float)
        elif self.big_type == self.CYLINDER:
            self.big_x, self.big_y, self.big_z = self._create_cylinder()
            self.big_mask = np.array([1, 1, 0], dtype=float)
        else:
            self.big_x, self.big_y, self.big_z = self._create_plane()
            self.big_mask = np.array([1, 0, 0], dtype=float)

    # Энергии и температура (в приведённых единицах k_B=1)
    def kinetic_energy(self):
        e_small = 0.5 * self.small_mass * np.sum(self.small_vel**2)
        e_big = 0.5 * self.big_mass * float(np.dot(self.big_vel, self.big_vel))
        return e_small + e_big

    def temperature_reduced(self):
        # T_r = (m * <v^2>) / 3, k_B = 1
        mean_v2 = np.mean(np.sum(self.small_vel**2, axis=1))
        return self.small_mass * mean_v2 / 3.0

    def big_speed(self):
        return float(np.linalg.norm(self.big_vel))

    # Один шаг интегрирования
    def step(self, steps=1):
        for _ in range(steps):
            self._step_once()

    def _step_once(self):
        dt = self.dt
        # Глобальный сдвиг
        self.small_pos += self.small_vel * dt
        self.big_pos += self.big_vel * dt

        # Столкновения "крупная–мелкие"
        delta_pos = (self.small_pos - self.big_pos) * self.big_mask
        distance2 = np.sum(delta_pos * delta_pos, axis=-1)
        collides = distance2 < (self.big_radius + self.small_radius) ** 2

        if np.any(collides):
            idx = np.where(collides)[0]
            distance = np.sqrt(distance2[idx])
            distance = np.where(distance < 1e-10, 1e-10, distance)
            normal = delta_pos[idx] / distance[:, None]
            rel = (self.small_vel[idx] - self.big_vel)
            scalar_prod = np.sum(rel * normal, axis=-1)
            hit = scalar_prod < 0.0

            if np.any(hit):
                hit_idx_arr = idx[hit]
                normal_arr = normal[hit]
                for normal_hit, hit_idx in zip(normal_arr, hit_idx_arr):
                    j = -2.0 * np.dot(self.small_vel[hit_idx] - self.big_vel, normal_hit) / (1.0 / self.small_mass + 1.0 / self.big_mass)
                    # Коррекция перекрытия
                    self.small_pos[hit_idx] = self.small_pos[hit_idx] - delta_pos[hit_idx] + normal_hit * (self.big_radius + self.small_radius + 1e-6)
                    # Импульсный обмен
                    self.small_vel[hit_idx] += (j * normal_hit) / self.small_mass
                    self.big_vel -= (j * normal_hit) / self.big_mass
                    self.collisions += 1

        # Столкновения со стенками
        for i in range(3):
            # Мелкие — верхняя стенка
            over = self.small_pos[:, i] > self.box_half[i] - self.small_radius
            if np.any(over):
                self.small_pos[over, i] = self.box_half[i] - self.small_radius
                self.small_vel[over, i] *= -1.0

            # Мелкие — нижняя стенка
            under = self.small_pos[:, i] < -self.box_half[i] + self.small_radius
            if np.any(under):
                self.small_pos[under, i] = -self.box_half[i] + self.small_radius
                self.small_vel[under, i] *= -1.0

            # Крупная
            if self.big_pos[i] > self.box_half[i] - self.big_radius:
                self.big_pos[i] = self.box_half[i] - self.big_radius
                self.big_vel[i] *= -1.0
            if self.big_pos[i] < -self.box_half[i] + self.big_radius:
                self.big_pos[i] = -self.box_half[i] + self.big_radius
                self.big_vel[i] *= -1.0

        self.time += dt


# --------------------------
# Виджеты страниц
# --------------------------

class TitlePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)

        # Эмблемы сверху
        logos = QHBoxLayout()
        left_logo = QLabel()
        left_logo.setPixmap(QPixmap(os.path.join(get_assets_dir(), "logo_left.png")))
        right_logo = QLabel()
        right_logo.setPixmap(QPixmap(os.path.join(get_assets_dir(), "logo_right.png")))
        for lbl in (left_logo, right_logo):
            lbl.setScaledContents(True)
            lbl.setMaximumSize(160, 160)
            logos.addWidget(lbl)
        v.addLayout(logos)

        title = QLabel("Столкновения частиц в коробке")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 36px; font-weight: 600;")
        v.addWidget(title)

        # Кнопки вертикально
        btns = QVBoxLayout()
        self.btn_presentation = QPushButton("Презентация")
        self.btn_theory = QPushButton("Теория по теме")
        self.btn_authors = QPushButton("Авторы")
        self.btn_exit = QPushButton("Выход")
        for b in (self.btn_presentation, self.btn_theory, self.btn_authors, self.btn_exit):
            b.setMinimumHeight(100)
            b.setMaximumWidth(500)
            b.setMinimumWidth(500)
            b.setStyleSheet("font-size: 24px; font-weight: 600;")
            btns.addWidget(b, alignment=Qt.AlignmentFlag.AlignHCenter)
        v.addLayout(btns)
        v.addStretch()


class TheoryPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Текстовая часть
        text = QLabel(
            "Теория по теме\n\n"
            "Моделируется движение N упругих частиц в прямоугольной области и их столкновения с крупным объектом "
            "(сфера/цилиндр/плоскость) и стенками. Применяется простая упругая модель обмена импульсом.\n\n"
            "В приведённых единицах (k_B = 1) температура оценивается как "
            "T_r = m · <v^2> / 3, кинетическая энергия E_k = 0.5 · m · Σ v_i^2 (сумма по всем частицам и по крупной).\n\n"
            "Единицы на осях: t (s_r), длина (m_r), скорость (m_r/s_r), энергия (E_r), температура (T_r).\n\n"
            "Рекомендуется переключать режимы скорости для обзора и для сбора статистики."
        )
        text.setWordWrap(True)
        text.setStyleSheet("font-size: 14px;")
        layout.addWidget(text)

        # Загрузка и прокрутка отсканированных страниц (если положить в assets/theory/*.png)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        v = QVBoxLayout(container)

        assets_theory = os.path.join(get_assets_dir(), "theory")
        if os.path.isdir(assets_theory):
            imgs = sorted([f for f in os.listdir(assets_theory) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            if imgs:
                for name in imgs:
                    lbl = QLabel()
                    pm = QPixmap(os.path.join(assets_theory, name))
                    lbl.setPixmap(pm)
                    lbl.setScaledContents(True)
                    lbl.setMinimumHeight(300)
                    v.addWidget(lbl)
        else:
            hint = QLabel("Положите сканы теории в папку assets/theory (PNG/JPG), они появятся здесь.")
            v.addWidget(hint)

        v.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.btn_back = QPushButton("Назад")
        self.btn_back.clicked.connect(lambda: parent.setCurrentIndex(0))
        layout.addWidget(self.btn_back, alignment=Qt.AlignmentFlag.AlignRight)


class AuthorsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Авторы и руководитель")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        # Блок авторов: загрузка из assets/authors
        grid = QHBoxLayout()
        for slot in range(3):
            pane = QVBoxLayout()
            img = QLabel()
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name = QLabel("Фамилия Имя\nГруппа ХХ-ХХ")
            name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            name.setStyleSheet("font-size: 14px;")
            img.setPixmap(self._load_author_image(slot))
            img.setScaledContents(True)
            img.setMinimumSize(160, 160)
            pane.addWidget(img)
            pane.addWidget(name)
            grid.addLayout(pane)
        layout.addLayout(grid)

        sup = QLabel("Преподаватель–руководитель: Фамилия Имя Отчество")
        sup.setStyleSheet("font-size: 14px;")
        layout.addWidget(sup)
        layout.addStretch(1)

        self.btn_back = QPushButton("Назад")
        self.btn_back.clicked.connect(lambda: parent.setCurrentIndex(0))
        layout.addWidget(self.btn_back, alignment=Qt.AlignmentFlag.AlignRight)

    def _load_author_image(self, idx):
        path = os.path.join(get_assets_dir(), "authors", f"author{idx+1}.png")
        if os.path.isfile(path):
            return QPixmap(path)
        return QPixmap()


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


class PresentationPage(QWidget):
    def __init__(self, model: SimulationModel, parent=None):
        super().__init__(parent)
        self.model = model

        # Виджеты
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.canvas3d = Mpl3DCanvas()
        self.canvas2d = Mpl2DCanvas()

        # 3D оси
        self._setup_3d_axes()

        # 2D оси
        self._setup_2d_axes()

        # Контролы
        controls = self._build_controls_panel()

        # Компоновка
        left_layout.addWidget(self.canvas3d)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.canvas2d)
        right_layout.addWidget(controls)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)

        # Таймер
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.steps_per_frame = 1  # медленный режим
        self.running = False

        # Данные для графиков
        self.t_hist = []
        self.E_hist = []
        self.T_hist = []
        self.Vb_hist = []
        self.max_hist_len = 2000

        # Инициализация 3D
        self._init_3d_artists()
        self._redraw_3d()

    # ---------- UI построение ----------

    def _build_controls_panel(self):
        panel = QGroupBox("Параметры и управление")
        form = QFormLayout(panel)

        # Параметры
        self.sb_N = QSpinBox()
        self.sb_N.setRange(1, 20000)
        self.sb_N.setValue(self.model.N)

        self.dsb_dt = QDoubleSpinBox()
        self.dsb_dt.setDecimals(5)
        self.dsb_dt.setRange(1e-5, 0.1)
        self.dsb_dt.setSingleStep(0.001)
        self.dsb_dt.setValue(self.model.dt)

        self.dsb_speed = QDoubleSpinBox()
        self.dsb_speed.setRange(0.0, 1000.0)
        self.dsb_speed.setSingleStep(1.0)
        self.dsb_speed.setValue(self.model.init_speed)

        self.dsb_small_r = QDoubleSpinBox()
        self.dsb_small_r.setRange(0.001, float(min(self.model.box_half) * 0.5))
        self.dsb_small_r.setSingleStep(0.01)
        self.dsb_small_r.setValue(self.model.small_radius)

        self.dsb_small_m = QDoubleSpinBox()
        self.dsb_small_m.setRange(1e-6, 1e6)
        self.dsb_small_m.setDecimals(6)
        self.dsb_small_m.setValue(self.model.small_mass)

        self.cb_big_type = QComboBox()
        self.cb_big_type.addItems(["Сфера", "Цилиндр", "Плоскость"])
        self.cb_big_type.setCurrentIndex(self.model.big_type)

        self.dsb_big_r = QDoubleSpinBox()
        self.dsb_big_r.setRange(0.001, float(min(self.model.box_half)))
        self.dsb_big_r.setSingleStep(0.05)
        self.dsb_big_r.setValue(self.model.big_radius)

        self.dsb_big_m = QDoubleSpinBox()
        self.dsb_big_m.setRange(1e-6, 1e9)
        self.dsb_big_m.setDecimals(6)
        self.dsb_big_m.setValue(self.model.big_mass)

        self.sb_big_res = QSpinBox()
        self.sb_big_res.setRange(6, 128)
        self.sb_big_res.setValue(self.model.big_resolution)

        # Режим скорости
        self.sb_steps = QSpinBox()
        self.sb_steps.setRange(1, 1000)
        self.sb_steps.setValue(1)
        self.sb_steps.setToolTip("Число шагов интегрирования на один кадр (1 — медленно, >1 — быстро)")

        # Кнопки
        self.btn_start = QPushButton("Старт")
        self.btn_pause = QPushButton("Пауза")
        self.btn_reset = QPushButton("Стоп/Сброс")
        self.btn_apply = QPushButton("Применить параметры")

        # Привязка событий
        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_apply.clicked.connect(self.apply_params)

        form.addRow("N (шт.)", self.sb_N)
        form.addRow("dt (s_r)", self.dsb_dt)
        form.addRow("Начальная скорость (m_r/s_r)", self.dsb_speed)
        form.addRow("Радиус мелкой (m_r)", self.dsb_small_r)
        form.addRow("Масса мелкой (kg_r)", self.dsb_small_m)
        form.addRow("Тип крупной", self.cb_big_type)
        form.addRow("Радиус крупной (m_r)", self.dsb_big_r)
        form.addRow("Масса крупной (kg_r)", self.dsb_big_m)
        form.addRow("Разрешение крупной (шт.)", self.sb_big_res)
        form.addRow("Шагов/кадр", self.sb_steps)
        form.addRow(self._buttons_row([self.btn_start, self.btn_pause, self.btn_reset, self.btn_apply]))
        return panel

    def _buttons_row(self, buttons):
        row = QWidget()
        h = QHBoxLayout(row)
        for b in buttons:
            h.addWidget(b)
        return row

    # ---------- 3D/2D оси и артисты ----------

    def _setup_3d_axes(self):
        ax = self.canvas3d.ax
        bh = self.model.box_half
        ax.set_xlim(-bh[0], bh[0])
        ax.set_ylim(-bh[1], bh[1])
        ax.set_zlim(-bh[2], bh[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x (m_r)")
        ax.set_ylabel("y (m_r)")
        ax.set_zlabel("z (m_r)")

    def _setup_2d_axes(self):
        ax1 = self.canvas2d.ax1
        ax2 = self.canvas2d.ax2

        ax1.set_xlabel("t (s_r)")
        ax1.set_ylabel("E_k (E_r)")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("t (s_r)")
        ax2.set_ylabel("T (T_r) / V_big (m_r/s_r)")
        ax2.grid(True, alpha=0.3)

        self.ln_E, = ax1.plot([], [], color="C3", label="E_k")
        ax1.legend(loc="upper right")

        self.ln_T, = ax2.plot([], [], color="C0", label="T")
        self.ln_Vb, = ax2.plot([], [], color="C2", label="V_big")
        ax2.legend(loc="upper right")

    def _init_3d_artists(self):
        ax = self.canvas3d.ax
        sp = self.model.small_pos
        # Точки мелких
        self.scatter = ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=20, c="C0", depthshade=True)
        # Поверхность крупной
        self.surface = ax.plot_surface(
            self.model.big_x + self.model.big_pos[0],
            self.model.big_y + self.model.big_pos[1],
            self.model.big_z + self.model.big_pos[2],
            color="r", alpha=0.9
        )

    def _redraw_3d(self):
        # Обновление точек
        sp = self.model.small_pos
        self.scatter._offsets3d = (sp[:, 0], sp[:, 1], sp[:, 2])
        # Обновление поверхности: удалить и создать заново (простая и надёжная стратегия)
        self.surface.remove()
        ax = self.canvas3d.ax
        self.surface = ax.plot_surface(
            self.model.big_x + self.model.big_pos[0],
            self.model.big_y + self.model.big_pos[1],
            self.model.big_z + self.model.big_pos[2],
            color="r", alpha=0.9
        )
        self.canvas3d.draw_idle()

    # ---------- Управление симуляцией ----------

    def start(self):
        if not self.running:
            self.steps_per_frame = max(1, int(self.sb_steps.value()))
            self.timer.start(16)  # ~60 FPS визуализации
            self.running = True

    def pause(self):
        if self.running:
            self.timer.stop()
            self.running = False

    def reset(self):
        # Останов, сброс, очистка графиков
        self.pause()
        self.model.reset()
        self.t_hist.clear()
        self.E_hist.clear()
        self.T_hist.clear()
        self.Vb_hist.clear()
        self.ln_E.set_data([], [])
        self.ln_T.set_data([], [])
        self.ln_Vb.set_data([], [])
        self.canvas2d.draw_idle()
        self._redraw_3d()

    def apply_params(self):
        # Применяем параметры, переинициализируем систему
        self.model.set_params(
            N=self.sb_N.value(),
            dt=self.dsb_dt.value(),
            init_speed=self.dsb_speed.value(),
            small_radius=self.dsb_small_r.value(),
            small_mass=self.dsb_small_m.value(),
            big_type=self.cb_big_type.currentIndex(),
            big_radius=self.dsb_big_r.value(),
            big_mass=self.dsb_big_m.value(),
            big_resolution=self.sb_big_res.value()
        )
        self.reset()

    def _on_tick(self):
        # Несколько шагов за кадр (быстрый сбор статистики)
        self.model.step(self.steps_per_frame)

        # Обновление 3D
        self._redraw_3d()

        # Обновление графиков
        t = self.model.time
        E = self.model.kinetic_energy()
        T = self.model.temperature_reduced()
        Vb = self.model.big_speed()
        self._append_hist(t, E, T, Vb)
        self._update_plots()

    def _append_hist(self, t, E, T, Vb):
        self.t_hist.append(t)
        self.E_hist.append(E)
        self.T_hist.append(T)
        self.Vb_hist.append(Vb)

        if len(self.t_hist) > self.max_hist_len:
            self.t_hist = self.t_hist[-self.max_hist_len:]
            self.E_hist = self.E_hist[-self.max_hist_len:]
            self.T_hist = self.T_hist[-self.max_hist_len:]
            self.Vb_hist = self.Vb_hist[-self.max_hist_len:]

    def _update_plots(self):
        self.ln_E.set_data(self.t_hist, self.E_hist)
        self.ln_T.set_data(self.t_hist, self.T_hist)
        self.ln_Vb.set_data(self.t_hist, self.Vb_hist)

        # Автомасштабирование по видимым данным
        for ax, ys in [(self.canvas2d.ax1, [self.E_hist]), (self.canvas2d.ax2, [self.T_hist, self.Vb_hist])]:
            ax.relim()
            ax.autoscale_view()

        self.canvas2d.draw_idle()


# --------------------------
# Главное окно (страницы)
# --------------------------

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

        # Навигация
        self.title.btn_presentation.clicked.connect(lambda: self.stack.setCurrentWidget(self.presentation))
        self.title.btn_theory.clicked.connect(lambda: self.stack.setCurrentWidget(self.theory))
        self.title.btn_authors.clicked.connect(lambda: self.stack.setCurrentWidget(self.authors))
        self.title.btn_exit.clicked.connect(self.close)

        # Начальная страница
        self.stack.setCurrentWidget(self.title)


# --------------------------
# Утилиты
# --------------------------

def get_base_dir():
    if getattr(sys, 'frozen', False):  # PyInstaller
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))

def get_assets_dir():
    # Ожидается папка assets рядом с exe/py
    base = get_base_dir()
    # Если запаковано — искать рядом с executable
    exedir = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(exedir, "assets"),
        os.path.join(base, "assets"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]  # путь по умолчанию

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
