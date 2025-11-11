import os
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QFormLayout, QScrollArea, QCheckBox
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ui.mpl_canvases import Mpl3DCanvas, Mpl2DCanvas
from utils.paths import get_assets_dir


class TitlePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)

        logos = QHBoxLayout()
        left_logo = QLabel(); left_logo.setPixmap(QPixmap(os.path.join(get_assets_dir(), "logo_left.png")))
        right_logo = QLabel(); right_logo.setPixmap(QPixmap(os.path.join(get_assets_dir(), "logo_right.png")))
        for lbl in (left_logo, right_logo):
            lbl.setScaledContents(True); lbl.setMaximumSize(160, 160); logos.addWidget(lbl)
        v.addLayout(logos)

        title = QLabel("Столкновения частиц в коробке")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 36px; font-weight: 600;")
        v.addWidget(title)

        btns = QVBoxLayout()
        self.btn_presentation = QPushButton("Презентация")
        self.btn_theory = QPushButton("Теория по теме")
        self.btn_authors = QPushButton("Авторы")
        self.btn_exit = QPushButton("Выход")
        for b in (self.btn_presentation, self.btn_theory, self.btn_authors, self.btn_exit):
            b.setMinimumHeight(100); b.setMaximumWidth(500); b.setMinimumWidth(500)
            b.setStyleSheet("font-size: 24px; font-weight: 600;"); btns.addWidget(b, alignment=Qt.AlignmentFlag.AlignHCenter)
        v.addLayout(btns)
        v.addStretch()


# --- замените существующий TheoryPage на этот ---
class TheoryPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from theory.content import SECTIONS
        from theory.render import latex_to_pixmap

        layout = QVBoxLayout(self)

        head = QLabel("Теория по теме")
        head.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(head)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); v = QVBoxLayout(container)

        for sec in SECTIONS:
            t = QLabel(sec["title"])
            t.setStyleSheet("font-size: 18px; font-weight: 600; margin-top: 8px;")
            v.addWidget(t)

            if sec.get("bullets"):
                for b in sec["bullets"]:
                    lb = QLabel("• " + b); lb.setWordWrap(True)
                    lb.setStyleSheet("font-size: 14px;")
                    v.addWidget(lb)

            if sec.get("formulas"):
                for tex, cap in sec["formulas"]:
                    pm = latex_to_pixmap(tex)
                    img = QLabel(); img.setPixmap(pm); img.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                    cap_w = QLabel(cap); cap_w.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                    cap_w.setStyleSheet("color:#555; font-size:12px; margin-bottom:6px;")
                    v.addWidget(img); v.addWidget(cap_w)

        v.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.btn_back = QPushButton("Назад"); self.btn_back.clicked.connect(lambda: parent.setCurrentIndex(0))
        layout.addWidget(self.btn_back, alignment=Qt.AlignmentFlag.AlignRight)


class AuthorsPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel("Авторы"); title.setStyleSheet("font-size: 36px; font-weight: 600;")
        layout.addWidget(title)
        hl = QHBoxLayout()
        for i in range(3):
            v = QVBoxLayout(); img = QLabel()
            path = os.path.join(get_assets_dir(), "authors", f"author{i+1}.png")
            if os.path.isfile(path):
                img.setPixmap(QPixmap(path))
            img.setScaledContents(True); img.setMaximumSize(160, 160)
            tag = QLabel(f"Автор {i+1}"); tag.setStyleSheet("font-size: 24px; font-weight: 600;"); tag.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            v.addWidget(img); v.addWidget(tag); hl.addLayout(v)
        layout.addLayout(hl)
        sup = QLabel("Руководитель: Имя Отчество Преподавателя"); sup.setStyleSheet("font-size: 36px; font-weight: 600;")
        layout.addWidget(sup)
        self.btn_back = QPushButton("Назад"); self.btn_back.clicked.connect(lambda: parent.setCurrentIndex(0))
        layout.addWidget(self.btn_back, alignment=Qt.AlignmentFlag.AlignRight)


class PresentationPage(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QSplitter

        self.model = model

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); left_layout = QVBoxLayout(left)
        right = QWidget(); right_layout = QVBoxLayout(right)

        self.canvas3d = Mpl3DCanvas(); self.canvas2d = Mpl2DCanvas()
        left_layout.addWidget(self.canvas3d)
        right_layout.addWidget(self.canvas2d)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(0, 2); splitter.setStretchFactor(1, 3)

        controls = self._build_controls_panel()
        right_layout.addWidget(controls)

        main_layout = QVBoxLayout(self); main_layout.addWidget(splitter)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.steps_per_frame = 1
        self.running = False

        self.t_hist = []; self.E_hist = []; self.T_hist = []; self.Vb_hist = []
        self.max_hist_len = 2000

        # throttle поверхности (как раньше)
        self._frame_id = 0
        self._surface_refresh_every = 4
        self._surface_kind = None
        self.surface = None
        self.surface_poly = None

        self._setup_3d_axes(); self._setup_2d_axes(); self._init_3d_artists(); self._redraw_3d()

    def _build_controls_panel(self):
        panel = QGroupBox("Параметры и управление"); form = QFormLayout(panel)

        # базовые
        self.sb_N = QSpinBox(); self.sb_N.setRange(1, 20000); self.sb_N.setValue(self.model.N)
        self.dsb_dt = QDoubleSpinBox(); self.dsb_dt.setDecimals(5); self.dsb_dt.setRange(1e-5, 0.1); self.dsb_dt.setSingleStep(0.001); self.dsb_dt.setValue(self.model.dt)
        self.dsb_speed = QDoubleSpinBox(); self.dsb_speed.setRange(0.0, 1000.0); self.dsb_speed.setSingleStep(1.0); self.dsb_speed.setValue(self.model.init_speed)
        self.dsb_small_r = QDoubleSpinBox(); self.dsb_small_r.setRange(0.001, float(min(self.model.box_half) * 0.5)); self.dsb_small_r.setSingleStep(0.01); self.dsb_small_r.setValue(self.model.small_radius)
        self.dsb_small_m = QDoubleSpinBox(); self.dsb_small_m.setRange(1e-6, 1e6); self.dsb_small_m.setDecimals(6); self.dsb_small_m.setValue(self.model.small_mass)
        self.cb_big_type = QComboBox(); self.cb_big_type.addItems(["Сфера", "Цилиндр", "Плоскость"]); self.cb_big_type.setCurrentIndex(self.model.big_type)
        self.dsb_big_r = QDoubleSpinBox(); self.dsb_big_r.setRange(0.001, float(min(self.model.box_half))); self.dsb_big_r.setSingleStep(0.05); self.dsb_big_r.setValue(self.model.big_radius)
        self.dsb_big_m = QDoubleSpinBox(); self.dsb_big_m.setRange(1e-6, 1e9); self.dsb_big_m.setDecimals(6); self.dsb_big_m.setValue(self.model.big_mass)
        self.sb_big_res = QSpinBox(); self.sb_big_res.setRange(6, 128); self.sb_big_res.setValue(self.model.big_resolution)
        self.sb_steps = QSpinBox(); self.sb_steps.setRange(1, 1000); self.sb_steps.setValue(1)

        # новые контролы физики
        self.cb_gas_coll = QCheckBox("Столкновения молекула–молекула")
        self.cb_gas_coll.setChecked(self.model.enable_gas_collisions)

        self.cb_maxwell = QCheckBox("Максвелл. инициализация")
        self.cb_maxwell.setChecked(self.model.use_maxwell_init)

        self.dsb_temp = QDoubleSpinBox(); self.dsb_temp.setRange(1e-6, 1e6); self.dsb_temp.setValue(self.model.temp_target)
        self.dsb_temp.setSingleStep(0.5)

        self.sb_E_log = QSpinBox(); self.sb_E_log.setRange(0, 100000); self.sb_E_log.setValue(self.model.energy_log_every)
        self.sb_E_log.setToolTip("Интервал логирования энергии (шаги). 0 — выкл.")

        # кнопки
        self.btn_start = QPushButton("Старт"); self.btn_pause = QPushButton("Пауза")
        self.btn_reset = QPushButton("Стоп/Сброс"); self.btn_apply = QPushButton("Применить параметры")
        self.btn_start.clicked.connect(self.start); self.btn_pause.clicked.connect(self.pause)
        self.btn_reset.clicked.connect(self.reset); self.btn_apply.clicked.connect(self.apply_params)

        # разметка
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
        form.addRow(self.cb_gas_coll)
        form.addRow(self.cb_maxwell)
        form.addRow("T для Максвелла (T_r)", self.dsb_temp)
        form.addRow("Энергия: лог каждые k шагов", self.sb_E_log)
        form.addRow(self._buttons_row([self.btn_start, self.btn_pause, self.btn_reset, self.btn_apply]))
        return panel

    def _buttons_row(self, buttons):
        row = QWidget(); h = QHBoxLayout(row)
        for b in buttons: h.addWidget(b)
        return row

    # ── axes & artists ───────────────────────────────────────────────────────
    def _setup_3d_axes(self):
        ax = self.canvas3d.ax; bh = self.model.box_half
        ax.set_xlim(-bh[0], bh[0]); ax.set_ylim(-bh[1], bh[1]); ax.set_zlim(-bh[2], bh[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x (m_r)"); ax.set_ylabel("y (m_r)"); ax.set_zlabel("z (m_r)")

    def _setup_2d_axes(self):
        ax1 = self.canvas2d.ax1; ax2 = self.canvas2d.ax2
        ax1.set_xlabel("t (s_r)"); ax1.set_ylabel("E_k (E_r)"); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("t (s_r)"); ax2.set_ylabel("T (T_r) / V_big (m_r/s_r)"); ax2.grid(True, alpha=0.3)
        self.ln_E, = ax1.plot([], [], color="C3", label="E_k"); ax1.legend(loc="upper right")
        self.ln_T, = ax2.plot([], [], color="C0", label="T"); self.ln_Vb, = ax2.plot([], [], color="C2", label="V_big"); ax2.legend(loc="upper right")

    def _init_3d_artists(self):
        ax = self.canvas3d.ax; sp = self.model.small_pos
        # точки мелких — каждый кадр
        self.scatter = ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=20, c="C0", depthshade=True)
        # поверхность крупной
        if self.model.big_type == self.model.PLANE:
            self._surface_kind = 'plane'
            self.surface_poly = self._make_plane_poly()
            ax.add_collection3d(self.surface_poly)
            self.surface = None
        else:
            self._surface_kind = 'surface'
            self.surface = ax.plot_surface(
                self.model.big_x + self.model.big_pos[0],
                self.model.big_y + self.model.big_pos[1],
                self.model.big_z + self.model.big_pos[2],
                color="r", alpha=0.9
            )
            self.surface_poly = None

    def _redraw_3d(self):
        sp = self.model.small_pos
        self.scatter._offsets3d = (sp[:, 0], sp[:, 1], sp[:, 2])
        ax = self.canvas3d.ax

        if self._surface_kind == 'plane':
            verts = self._plane_vertices_world()
            self.surface_poly.set_verts([verts])
        else:
            self._frame_id += 1
            if (self._frame_id % self._surface_refresh_every) == 0:
                if self.surface is not None:
                    try:
                        self.surface.remove()
                    except Exception:
                        pass
                self.surface = ax.plot_surface(
                    self.model.big_x + self.model.big_pos[0],
                    self.model.big_y + self.model.big_pos[1],
                    self.model.big_z + self.model.big_pos[2],
                    color="r", alpha=0.9
                )
        self.canvas3d.draw_idle()

    # ── helpers для плоскости ────────────────────────────────────────────────
    def _make_plane_poly(self) -> Poly3DCollection:
        verts = self._plane_vertices_world()
        return Poly3DCollection([verts], alpha=0.9, facecolor="r", edgecolor="none")

    def _plane_vertices_world(self):
        x = self.model.big_pos[0]
        y0, y1 = -self.model.box_half[1], self.model.box_half[1]
        z0, z1 = -self.model.box_half[2], self.model.box_half[2]
        return [
            (x, y0, z0),
            (x, y1, z0),
            (x, y1, z1),
            (x, y0, z1),
        ]

    # ── controls handlers ────────────────────────────────────────────────────
    def start(self):
        if not self.running:
            self.steps_per_frame = max(1, int(self.sb_steps.value()))
            self.timer.start(16)
            self.running = True

    def pause(self):
        if self.running:
            self.timer.stop(); self.running = False

    def reset(self):
        self.pause(); self.model.reset()
        self.t_hist.clear(); self.E_hist.clear(); self.T_hist.clear(); self.Vb_hist.clear()
        self.ln_E.set_data([], []); self.ln_T.set_data([], []); self.ln_Vb.set_data([], [])
        self.canvas2d.draw_idle()

        # убрать старые артисты поверхности
        if self.surface_poly is not None:
            try:
                self.surface_poly.remove()
            except Exception:
                pass
            self.surface_poly = None
        if self.surface is not None:
            try:
                self.surface.remove()
            except Exception:
                pass
            self.surface = None

        # пересоздать артисты с учётом типа
        self._init_3d_artists()
        self._redraw_3d()

    def apply_params(self):
        self.model.set_params(
            N=self.sb_N.value(), dt=self.dsb_dt.value(), init_speed=self.dsb_speed.value(),
            small_radius=self.dsb_small_r.value(), small_mass=self.dsb_small_m.value(),
            big_type=self.cb_big_type.currentIndex(), big_radius=self.dsb_big_r.value(),
            big_mass=self.dsb_big_m.value(), big_resolution=self.sb_big_res.value(),
            enable_gas_collisions=self.cb_gas_coll.isChecked(),
            use_maxwell_init=self.cb_maxwell.isChecked(),
            temp_target=self.dsb_temp.value(),
            energy_log_every=self.sb_E_log.value()
        )
        self.reset()

    # ── tick ─────────────────────────────────────────────────────────────────
    def _on_tick(self):
        self.model.step(self.steps_per_frame)
        self._redraw_3d()
        t = self.model.time; E = self.model.kinetic_energy(); T = self.model.temperature_reduced(); Vb = self.model.big_speed()
        self._append_hist(t, E, T, Vb); self._update_plots()

    def _append_hist(self, t, E, T, Vb):
        self.t_hist.append(t); self.E_hist.append(E); self.T_hist.append(T); self.Vb_hist.append(Vb)
        if len(self.t_hist) > self.max_hist_len:
            self.t_hist = self.t_hist[-self.max_hist_len:]
            self.E_hist = self.E_hist[-self.max_hist_len:]
            self.T_hist = self.T_hist[-self.max_hist_len:]
            self.Vb_hist = self.Vb_hist[-self.max_hist_len:]

    def _update_plots(self):
        self.ln_E.set_data(self.t_hist, self.E_hist)
        self.ln_T.set_data(self.t_hist, self.T_hist)
        self.ln_Vb.set_data(self.t_hist, self.Vb_hist)
        for ax in (self.canvas2d.ax1, self.canvas2d.ax2):
            ax.relim(); ax.autoscale_view()
        self.canvas2d.draw_idle()
