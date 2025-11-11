import numpy as np


class SimulationModel:
    SPHERE, CYLINDER, PLANE = 0, 1, 2

    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)

        self.box_half = np.array([5.0, 5.0, 5.0], dtype=float)

        self.N = 100
        self.small_radius = 0.1
        self.small_mass = 1.0

        self.big_type = self.SPHERE
        self.big_radius = 1.0
        self.big_mass = 10.0
        self.big_resolution = 12

        self.dt = 0.01
        self.init_speed = 20.0

        self.big_pos = np.zeros(3, dtype=float)
        self.big_vel = np.zeros(3, dtype=float)
        self.small_pos = None
        self.small_vel = None

        self.big_mask = np.array([1.0, 1.0, 1.0], dtype=float)
        self.big_x = None
        self.big_y = None
        self.big_z = None

        self._init_particles()
        self._update_big_geometry()

        self.time = 0.0
        self.collisions = 0

    # ── initialization ───────────────────────────────────────────────────────
    def _init_particles(self):
        # равномерно в коробке
        self.small_pos = (self.rng.random((self.N, 3)) - 0.5) * (2 * self.box_half - 2 * self.small_radius)

        # нормальные компоненты, затем масштабируем среднюю скорость
        v = self.rng.normal(size=(self.N, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        v *= self.init_speed
        self.small_vel = v

        self.big_pos[:] = 0.0
        self.big_vel[:] = 0.0
        self.time = 0.0
        self.collisions = 0

        # избегаем старта с пересечениями с крупной фигурой
        self._push_out_small_inside_big()

    def _push_out_small_inside_big(self):
        # мягко выталкиваем тех, кто оказался внутри крупной фигуры
        delta = (self.small_pos - self.big_pos) * self.big_mask
        d2 = np.sum(delta * delta, axis=1)
        rad2 = (self.big_radius + self.small_radius) ** 2
        inside = d2 < rad2
        if np.any(inside):
            d = np.sqrt(d2[inside])
            d = np.where(d < 1e-10, 1e-10, d)
            n = delta[inside] / d[:, None]
            self.small_pos[inside] = self.big_pos + n * (self.big_radius + self.small_radius + 1e-6)

    # ── public params ────────────────────────────────────────────────────────
    def set_params(self, N=None, dt=None, init_speed=None, small_radius=None, small_mass=None,
                   big_type=None, big_radius=None, big_mass=None, big_resolution=None):
        if N is not None:
            self.N = int(np.clip(N, 1, 20000))
        if dt is not None:
            self.dt = float(np.clip(dt, 1e-5, 0.1))
        if init_speed is not None:
            self.init_speed = float(np.clip(init_speed, 0.0, 1000.0))
        if small_radius is not None:
            self.small_radius = float(np.clip(small_radius, 1e-3, float(min(self.box_half) * 0.5)))
        if small_mass is not None:
            self.small_mass = float(np.clip(small_mass, 1e-6, 1e6))
        if big_type is not None:
            self.big_type = int(big_type)
        if big_radius is not None:
            self.big_radius = float(np.clip(big_radius, 1e-3, float(min(self.box_half))))
        if big_mass is not None:
            self.big_mass = float(np.clip(big_mass, 1e-6, 1e9))
        if big_resolution is not None:
            self.big_resolution = int(np.clip(big_resolution, 6, 128))

        self._update_big_geometry()

    def reset(self):
        self._init_particles()

    # ── geometry ─────────────────────────────────────────────────────────────
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
        # плита в плоскости YZ
        y = np.linspace(-self.box_half[1], self.box_half[1], 2)
        z = np.linspace(-self.box_half[2], self.box_half[2], 2)
        y, z = np.meshgrid(y, z)
        x = np.zeros_like(y)
        return x, y, z

    def _update_big_geometry(self):
        if self.big_type == self.SPHERE:
            self.big_x, self.big_y, self.big_z = self._create_sphere()
            self.big_mask = np.array([1.0, 1.0, 1.0])
        elif self.big_type == self.CYLINDER:
            self.big_x, self.big_y, self.big_z = self._create_cylinder()
            self.big_mask = np.array([1.0, 1.0, 0.0])
        else:
            self.big_x, self.big_y, self.big_z = self._create_plane()
            self.big_mask = np.array([1.0, 0.0, 0.0])

    # ── thermodynamics helpers ───────────────────────────────────────────────
    def kinetic_energy(self):
        e_small = 0.5 * self.small_mass * np.sum(self.small_vel ** 2)
        e_big = 0.5 * self.big_mass * float(np.dot(self.big_vel, self.big_vel))
        return e_small + e_big

    def temperature_reduced(self, f_small=3):
        mean_v2 = np.mean(np.sum(self.small_vel ** 2, axis=1))
        return self.small_mass * mean_v2 / float(f_small)

    def big_speed(self):
        return float(np.linalg.norm(self.big_vel))

    def big_energy_components(self):
        v = self.big_vel
        m = self.big_mass
        return 0.5 * m * v[0] ** 2, 0.5 * m * v[1] ** 2, 0.5 * m * v[2] ** 2

    # ── dof helpers ──────────────────────────────────────────────────────────
    def _enforce_big_dofs(self):
        # обнуляем запрещённые степени свободы
        self.big_vel *= self.big_mask
        self.big_pos *= self.big_mask

    def _big_extent_per_axis(self):
        if self.big_type == self.SPHERE:
            return np.array([self.big_radius, self.big_radius, self.big_radius])
        if self.big_type == self.CYLINDER:
            return np.array([self.big_radius, self.big_radius, 0.0])
        return np.array([self.big_radius, 0.0, 0.0])  # плита

    # ── integration step ─────────────────────────────────────────────────────
    def step(self, steps=1):
        for _ in range(steps):
            self._step_once()

    def _step_once(self):
        dt = self.dt

        # перед шагом уважим размерность движения крупной
        self._enforce_big_dofs()

        self.small_pos += self.small_vel * dt
        self.big_pos += self.big_vel * dt

        # столкновения с крупной
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
                    j = -2.0 * np.dot(self.small_vel[hit_idx] - self.big_vel, normal_hit) / \
                        (1.0 / self.small_mass + 1.0 / self.big_mass)
                    # коррекция перекрытия
                    self.small_pos[hit_idx] = self.small_pos[hit_idx] - (self.small_pos[hit_idx] - self.big_pos) * self.big_mask + \
                                              normal_hit * (self.big_radius + self.small_radius + 1e-6)
                    # обмен импульсом
                    self.small_vel[hit_idx] += (j * normal_hit) / self.small_mass
                    self.big_vel -= (j * normal_hit) / self.big_mass
                    self.collisions += 1

        # стены
        for i in range(3):
            over = self.small_pos[:, i] > self.box_half[i] - self.small_radius
            if np.any(over):
                self.small_pos[over, i] = self.box_half[i] - self.small_radius
                self.small_vel[over, i] *= -1.0

            under = self.small_pos[:, i] < -self.box_half[i] + self.small_radius
            if np.any(under):
                self.small_pos[under, i] = -self.box_half[i] + self.small_radius
                self.small_vel[under, i] *= -1.0

        # стены для крупной с геометрическим экстентом
        ext = self._big_extent_per_axis()
        for i in range(3):
            if self.big_pos[i] > self.box_half[i] - ext[i]:
                self.big_pos[i] = self.box_half[i] - ext[i]
                self.big_vel[i] *= -1.0
            if self.big_pos[i] < -self.box_half[i] + ext[i]:
                self.big_pos[i] = -self.box_half[i] + ext[i]
                self.big_vel[i] *= -1.0

        self.time += dt