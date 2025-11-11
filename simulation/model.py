# simulation/model.py
import numpy as np

class SimulationModel:
    SPHERE, CYLINDER, PLANE = 0, 1, 2

    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)

        # Коробка (полудиагонали)
        self.box_half = np.array([5.0, 5.0, 5.0], dtype=float)

        # Параметры газа
        self.N = 100
        self.small_radius = 0.1
        self.small_mass = 1.0

        # Крупный объект
        self.big_type = self.SPHERE
        self.big_radius = 1.0
        self.big_mass = 10.0
        self.big_resolution = 12

        # Интегрирование
        self.dt = 0.01
        self.init_speed = 20.0   # используется, если use_maxwell_init=False

        # Новые физпараметры
        self.enable_gas_collisions = True     # столкновения молекула–молекула
        self.use_maxwell_init = False         # Максвелл-Больцман инициализация
        self.temp_target = 5.0                # T (k_B = 1)
        self.energy_log_every = 0             # 0 = выкл, >0 шагов между логами

        # Состояние
        self.big_pos = np.zeros(3, dtype=float)
        self.big_vel = np.zeros(3, dtype=float)
        self.small_pos = None
        self.small_vel = None

        # Геометрия крупной
        self.big_mask = np.array([1, 1, 1], dtype=float)
        self.big_x = None
        self.big_y = None
        self.big_z = None

        # Инициализация
        self._init_particles()
        self._update_big_geometry()

        # Учёт времени и счётчики
        self.time = 0.0
        self.collisions = 0
        self._steps = 0
        self._E0 = self.kinetic_energy()

    # ---------- Параметры ----------
    def set_params(self, N=None, dt=None, init_speed=None, small_radius=None, small_mass=None,
                   big_type=None, big_radius=None, big_mass=None, big_resolution=None,
                   enable_gas_collisions=None, use_maxwell_init=None, temp_target=None,
                   energy_log_every=None):
        if N is not None:            self.N = int(np.clip(N, 1, 20000))
        if dt is not None:           self.dt = float(np.clip(dt, 1e-5, 0.1))
        if init_speed is not None:   self.init_speed = float(np.clip(init_speed, 0.0, 1000.0))
        if small_radius is not None: self.small_radius = float(np.clip(small_radius, 1e-3, min(self.box_half) * 0.5))
        if small_mass is not None:   self.small_mass = float(np.clip(small_mass, 1e-6, 1e6))
        if big_type is not None:     self.big_type = int(big_type)
        if big_radius is not None:   self.big_radius = float(np.clip(big_radius, 1e-3, min(self.box_half)))
        if big_mass is not None:     self.big_mass = float(np.clip(big_mass, 1e-6, 1e9))
        if big_resolution is not None: self.big_resolution = int(np.clip(big_resolution, 6, 128))
        if enable_gas_collisions is not None: self.enable_gas_collisions = bool(enable_gas_collisions)
        if use_maxwell_init is not None:      self.use_maxwell_init = bool(use_maxwell_init)
        if temp_target is not None:           self.temp_target = float(np.clip(temp_target, 1e-6, 1e6))
        if energy_log_every is not None:      self.energy_log_every = int(max(0, energy_log_every))
        self._update_big_geometry()

    def reset(self):
        self._init_particles()
        self.time = 0.0
        self.collisions = 0
        self._steps = 0
        self._E0 = self.kinetic_energy()

    # ---------- Инициализация ----------
    def _init_particles(self):
        # позиции равномерно, без выхода за стенки
        span = (2 * self.box_half - 2 * self.small_radius)
        self.small_pos = (self.rng.random((self.N, 3)) - 0.5) * span

        if self.use_maxwell_init:
            # vx, vy, vz ~ N(0, sigma^2), где sigma^2 = kT/m; k_B=1
            sigma = np.sqrt(self.temp_target / self.small_mass)
            v = self.rng.normal(loc=0.0, scale=sigma, size=(self.N, 3))
        else:
            # случайные направления с фиксированным модулем init_speed
            v = self.rng.normal(size=(self.N, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            v *= self.init_speed

        self.small_vel = v
        self.big_pos[:] = 0.0
        self.big_vel[:] = 0.0

        # выталкивание частиц, оказавшихся внутри крупного тела
        self._push_out_small_inside_big()

    def _push_out_small_inside_big(self):
        # если мелкая внутри крупной — выталкиваем на поверхность по нормали
        dp = (self.small_pos - self.big_pos) * self._current_mask()
        r_eff = self.big_radius + self.small_radius
        d2 = np.sum(dp * dp, axis=1)
        inside = d2 < (r_eff ** 2)
        if not np.any(inside):
            return
        d = np.sqrt(d2[inside])
        d = np.where(d < 1e-10, 1e-10, d)
        n = dp[inside] / d[:, None]
        self.small_pos[inside] = self.big_pos + n * (r_eff + 1e-6)

    # ---------- Геометрия крупной ----------
    def _create_sphere(self):
        phi = np.linspace(0, 2*np.pi, self.big_resolution)
        theta = np.linspace(np.pi, 0, max(self.big_resolution // 2, 3))
        phi, theta = np.meshgrid(phi, theta)
        x = self.big_radius * np.cos(phi) * np.sin(theta)
        y = self.big_radius * np.sin(phi) * np.sin(theta)
        z = self.big_radius * np.cos(theta)
        return x, y, z

    def _create_cylinder(self):
        theta = np.linspace(0, 2*np.pi, self.big_resolution)
        z = np.linspace(-self.box_half[-1], self.box_half[-1], 2)
        theta, z = np.meshgrid(theta, z)
        x = self.big_radius * np.cos(theta)
        y = self.big_radius * np.sin(theta)
        return x, y, z

    def _create_plane(self):
        x = np.array([[1,1,1,1,1],[0,0,0,0,0]], dtype=float) * self.big_radius * 2 - self.big_radius
        y = np.array([[0,0,1,1,0],[0,0,1,1,0]], dtype=float) * self.box_half[1] * 2 - self.box_half[1]
        z = np.array([[0,1,1,0,0],[0,1,1,0,0]], dtype=float) * self.box_half[2] * 2 - self.box_half[2]
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

    def _current_mask(self):
        return self.big_mask

    # ---------- Энергии, температура ----------
    def kinetic_energy(self):
        e_small = 0.5 * self.small_mass * np.sum(self.small_vel**2)
        e_big = 0.5 * self.big_mass * float(np.dot(self.big_vel, self.big_vel))
        return e_small + e_big

    def temperature_reduced(self):
        # k_B = 1, T = m * <v^2> / 3
        mean_v2 = np.mean(np.sum(self.small_vel**2, axis=1))
        return self.small_mass * mean_v2 / 3.0

    def temperature_tensor(self):
        # компоненты "анизотропной" температуры по осям: T_i = m * <v_i^2>
        comp_mean = np.mean(self.small_vel**2, axis=0)  # <vx^2>, <vy^2>, <vz^2>
        return self.small_mass * comp_mean  # массив длины 3

    def big_speed(self):
        return float(np.linalg.norm(self.big_vel))

    # ---------- Шаг интегрирования ----------
    def step(self, steps=1):
        for _ in range(steps):
            self._step_once()

    def _step_once(self):
        dt = self.dt
        # 1) свободный ход
        self.small_pos += self.small_vel * dt
        self.big_pos += self.big_vel * dt

        # 2) столкновения мелкая–крупная
        self._collide_small_big()

        # 3) столкновения со стенками
        self._collide_walls()

        # 4) столкновения мелкая–мелкая (новое)
        if self.enable_gas_collisions:
            self._collide_small_small()

        # Счётчики
        self.time += dt
        self._steps += 1

        # Контроль энергии (опционально)
        if self.energy_log_every and (self._steps % self.energy_log_every == 0):
            dE = self.kinetic_energy() - self._E0
            # печать в консоль; при желании заменить на сбор в лог
            print(f"[energy] t={self.time:.3f}  ΔE={dE:.6e}")

    # ---------- Столкновения ----------
    def _collide_small_big(self):
        dp = (self.small_pos - self.big_pos) * self._current_mask()
        r_eff = self.big_radius + self.small_radius
        d2 = np.sum(dp * dp, axis=1)
        collides = d2 < (r_eff ** 2)
        if not np.any(collides):
            return

        idx = np.where(collides)[0]
        d = np.sqrt(d2[idx])
        d = np.where(d < 1e-10, 1e-10, d)
        n = dp[idx] / d[:, None]
        rel = (self.small_vel[idx] - self.big_vel)
        s = np.sum(rel * n, axis=1)  # проекция относительной скорости

        hit = s < 0.0
        if not np.any(hit):
            return

        hit_idx = idx[hit]
        n_hit = n[hit]
        for p, i in zip(n_hit, hit_idx):
            # коррекция перекрытия: выталкивание мелкой на поверхность
            self.small_pos[i] = self.big_pos + p * (r_eff + 1e-6)
            # импульс (упругое e=1)
            j = -2.0 * np.dot(self.small_vel[i] - self.big_vel, p) / (1.0 / self.small_mass + 1.0 / self.big_mass)
            self.small_vel[i] += (j * p) / self.small_mass
            self.big_vel      -= (j * p) / self.big_mass
            self.collisions += 1

    def _collide_walls(self):
        # мелкие
        for i in range(3):
            over = self.small_pos[:, i] > self.box_half[i] - self.small_radius
            if np.any(over):
                self.small_pos[over, i] = self.box_half[i] - self.small_radius
                self.small_vel[over, i] *= -1.0
            under = self.small_pos[:, i] < -self.box_half[i] + self.small_radius
            if np.any(under):
                self.small_pos[under, i] = -self.box_half[i] + self.small_radius
                self.small_vel[under, i] *= -1.0

        # крупная
        for i in range(3):
            if self.big_pos[i] > self.box_half[i] - self.big_radius:
                self.big_pos[i] = self.box_half[i] - self.big_radius
                self.big_vel[i] *= -1.0
            if self.big_pos[i] < -self.box_half[i] + self.big_radius:
                self.big_pos[i] = -self.box_half[i] + self.big_radius
                self.big_vel[i] *= -1.0

    def _collide_small_small(self):
        """Упругие столкновения одинаковых шаров (m=r=константы).
           O(N^2) — достаточно быстро для N~100.
        """
        N = self.N
        r = self.small_radius
        m = self.small_mass
        pos = self.small_pos
        vel = self.small_vel
        rr = 2.0 * r
        rr2 = rr * rr

        # двойной цикл i<j
        for i in range(N - 1):
            # векторно на кусках можно ускорить, но N невелик
            pi = pos[i]
            vi = vel[i]
            dp = pos[i+1:] - pi
            d2 = np.einsum('ij,ij->i', dp, dp)
            cand = np.where(d2 < rr2)[0]
            if cand.size == 0:
                continue

            for k in cand:
                j = i + 1 + k
                n_vec = pos[j] - pos[i]
                dist = np.linalg.norm(n_vec)
                if dist < 1e-12:
                    # случай почти совпадения, сгенерируем нормаль
                    n = self.rng.normal(size=3)
                    n /= np.linalg.norm(n)
                else:
                    n = n_vec / dist

                # проверка на сближение
                rel = vel[i] - vel[j]
                s = np.dot(rel, n)
                if s >= 0.0:
                    # расходятся или касание
                    continue

                # коррекция перекрытия поровну
                overlap = rr - dist
                if overlap > 0.0:
                    corr = 0.5 * (overlap + 1e-6)
                    pos[i] -= n * corr
                    pos[j] += n * corr

                # упругий обмен импульсом (одинаковые массы)
                jimp = -2.0 * s / (1.0/m + 1.0/m)  # = -s * m
                vel[i] += (jimp * n) / m
                vel[j] -= (jimp * n) / m
                self.collisions += 1
