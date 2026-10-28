#!/usr/bin/env python3
"""
Intelligent Vision-Based Object Tracking and Robot Control Simulator
====================================================================
(with vision-based obstacle detection and avoidance)

Pipeline (runs every frame, 30 Hz):

    Virtual world --> Pinhole camera render --> Perception
                                                 |-- ball detector   (red + round)     --> Kalman tracker --> PID follow control --+
                                                 |-- obstacle detector (everything else) --> obstacle map --> avoidance ----------+--> Robot
                                                                                                                                   |
                 robot moves, so the camera view changes <-------------------------------------------------------------------------+

Key idea: the BALL detector and the OBSTACLE detector are separate. The ball detector
only accepts blobs that are red AND round, so obstacles (even a red barrel) are never
mistaken for the target. Obstacles are found from colour/saturation, ranged with
monocular ground-plane geometry, stored in a small map, and avoided with a reactive
steering rule while the robot keeps following the ball.

Run:
    python simulator.py                 # headless, writes video + plots to ./output
    python simulator.py --live          # real-time window (press q to quit)
    python simulator.py --duration 60
"""
import argparse
import math
import os
import shutil
import subprocess

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
IMG_W, IMG_H = 320, 240            # camera resolution
FOV_DEG = 70.0                     # horizontal field of view
F_PX = (IMG_W / 2) / math.tan(math.radians(FOV_DEG / 2))   # focal length (pixels)
CX, CY = IMG_W / 2, IMG_H / 2      # principal point
CAM_H = 0.25                       # camera height above floor (m)
BALL_R = 0.15                      # target ball radius (m)
ARENA = 10.0                       # arena is ARENA x ARENA metres
DT = 1 / 30.0                      # simulation step

D_TARGET = 2.0                     # desired following distance (m)
MAX_COAST = 45                     # frames (1.5 s) the tracker may predict blindly
OCCLUSIONS = [(12.0, 13.0), (26.0, 26.8)]   # seconds when a board blocks the view

R_ROBOT = 0.28                     # robot body radius (m)

# Avoidance parameters
LOOK = 1.6           # how far ahead (m) the robot checks each candidate heading for obstacles
K_H = 3.0            # heading gain used while steering around an obstacle (rad/s per rad)
W_BLOCK = 3.0        # cost of a heading that runs into an obstacle
W_HYST = 0.4         # cost of changing the previously chosen heading (stops dithering)
INFLATE = 0.12       # extra clearance added around every obstacle (m)
MIN_SIGHTINGS = 20   # an obstacle must be seen this many frames before it is trusted
MAX_OBS_RANGE = 3.2  # ignore obstacle blobs further than this (depth too noisy)

# Scenery: slim pillars around the arena edge (x, y, radius, height, BGR colour)
SCENERY = [
    (0.5, 2.0, 0.2, 1.0, (200, 90, 30)), (0.5, 5.0, 0.2, 1.0, (60, 170, 60)), (0.5, 8.0, 0.2, 1.0, (30, 200, 220)),
    (9.5, 2.0, 0.2, 1.0, (60, 170, 60)), (9.5, 5.0, 0.2, 1.0, (200, 90, 30)), (9.5, 8.0, 0.2, 1.0, (30, 200, 220)),
    (2.5, 0.5, 0.2, 1.0, (30, 200, 220)), (5.0, 0.5, 0.2, 1.0, (200, 90, 30)), (7.5, 0.5, 0.2, 1.0, (60, 170, 60)),
    (2.5, 9.5, 0.2, 1.0, (60, 170, 60)), (5.0, 9.5, 0.2, 1.0, (30, 200, 220)), (7.5, 9.5, 0.2, 1.0, (200, 90, 30)),
    (8.6, 5.5, 0.2, 1.0, (40, 40, 215)),      # tall RED pillar: same colour as the ball, different shape
]
# Obstacles placed in the middle of the arena, right where the robot wants to drive.
OBSTACLES = [
    (4.0, 6.6, 0.30, 0.9, (40, 40, 215)),     # red barrel  -> must NOT be mistaken for the ball
    (3.0, 4.9, 0.35, 0.8, (150, 50, 130)),    # purple drum
    (4.5, 3.6, 0.30, 1.0, (0, 165, 255)),     # orange cylinder
    (4.6, 5.6, 0.30, 0.9, (170, 160, 20)),    # teal drum
]
ALL_OBJECTS = SCENERY + OBSTACLES
WORLD_CIRCLES = [(x, y, r) for x, y, r, _, _ in ALL_OBJECTS]


# --------------------------------------------------------------------------
# World objects
# --------------------------------------------------------------------------
class Target:
    """Ball that moves on a smooth Lissajous-like path (analytic, so ground truth is exact)."""

    def __init__(self):
        self.t = 0.0
        self.x, self.y = self._pos(0.0)

    @staticmethod
    def _pos(t):
        x = 5 + 3.0 * math.sin(0.18 * t) + 0.8 * math.sin(0.5 * t + 0.5)
        y = 5 + 3.0 * math.sin(0.13 * t + 1.0) + 0.8 * math.cos(0.41 * t)
        return x, y

    def step(self, dt):
        self.t += dt
        self.x, self.y = self._pos(self.t)


def check_obstacles(duration=40.0):
    """Sanity check: the ball's path must not run through an obstacle."""
    ts = np.arange(0, duration, 0.05)
    path = np.array([Target._pos(t) for t in ts])
    ok = True
    for x, y, r, _, _ in OBSTACLES:
        d = np.min(np.hypot(path[:, 0] - x, path[:, 1] - y))
        if d < r + BALL_R + 0.15:
            print(f"WARNING: obstacle at ({x},{y}) is only {d:.2f} m from the ball's path")
            ok = False
    return ok


class Robot:
    """Differential-drive (unicycle) robot with first-order actuator lag and solid-body collisions."""

    def __init__(self, x=1.5, y=1.5, th=0.3):
        self.x, self.y, self.th = x, y, th
        self.v = 0.0
        self.w = 0.0
        self.tau = 0.15           # actuator time constant (s)
        self.v_max, self.v_min, self.w_max = 1.8, -0.8, 2.5

    def step(self, v_cmd, w_cmd, dt):
        """Integrate motion. Returns True if the robot touched an obstacle this step."""
        v_cmd = float(np.clip(v_cmd, self.v_min, self.v_max))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))
        a = dt / (self.tau + dt)
        self.v += a * (v_cmd - self.v)
        self.w += a * (w_cmd - self.w)
        self.th += self.w * dt
        self.x += self.v * math.cos(self.th) * dt
        self.y += self.v * math.sin(self.th) * dt
        self.x = float(np.clip(self.x, 0.3, ARENA - 0.3))
        self.y = float(np.clip(self.y, 0.3, ARENA - 0.3))
        hit = False
        for cx, cy, r in WORLD_CIRCLES:                  # solid obstacles: push the robot out
            d = math.hypot(self.x - cx, self.y - cy)
            if d < r + R_ROBOT:
                hit = True
                nx, ny = (self.x - cx) / (d + 1e-9), (self.y - cy) / (d + 1e-9)
                self.x, self.y = cx + nx * (r + R_ROBOT), cy + ny * (r + R_ROBOT)
                self.v *= 0.3
        return hit


# --------------------------------------------------------------------------
# Camera simulation
# --------------------------------------------------------------------------
def to_robot_frame(robot, px, py):
    dx, dy = px - robot.x, py - robot.y
    c, s = math.cos(robot.th), math.sin(robot.th)
    return c * dx + s * dy, -s * dx + c * dy     # (forward, left)


def project(robot, px, py, h):
    """Pinhole projection of a world point at height h. Returns (u, v, depth) or None."""
    xf, yl = to_robot_frame(robot, px, py)
    if xf < 0.15:
        return None
    return CX - F_PX * yl / xf, CY + F_PX * (CAM_H - h) / xf, xf


def render_camera(robot, target, t, rng):
    """Synthesise the robot's camera image (BGR) and return (image, true ball (u,v,r) or None)."""
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    hz = int(CY)
    img[:hz] = (215, 190, 160)                                  # sky
    grad = np.linspace(0, 1, IMG_H - hz)[:, None]
    img[hz:] = np.stack([95 + 45 * grad, 105 + 45 * grad, 110 + 45 * grad], axis=2).astype(np.uint8)

    items = []                                                  # (depth, kind, data)
    for ox, oy, r, h, col in ALL_OBJECTS:
        p = project(robot, ox, oy, 0.0)
        if p and p[2] > 0.3:
            items.append((p[2], "obj", (p, r, h, col)))
    tp = project(robot, target.x, target.y, BALL_R)
    true_ball = None
    if tp:
        items.append((tp[2], "ball", tp))
        r = F_PX * BALL_R / tp[2]
        if -r < tp[0] < IMG_W + r and 0 < tp[1] < IMG_H:
            true_ball = (tp[0], tp[1], r)

    def ic(v):
        return int(np.clip(v, -5000, 5000))

    for _, kind, data in sorted(items, key=lambda i: -i[0]):
        if kind == "obj":
            (u, _, d), r, h, col = data
            hw = F_PX * r / d
            dn = max(d - r, 0.2)                                # near surface touches the floor
            bottom = CY + F_PX * CAM_H / dn
            top = CY - F_PX * (h - CAM_H) / dn
            x0, x1 = ic(u - hw), ic(u + hw)
            cv2.rectangle(img, (x0, ic(top)), (x1, ic(bottom)), col, -1)
            shade = tuple(int(c * 0.72) for c in col)           # darker right side -> cylinder look
            cv2.rectangle(img, (ic(u + hw * 0.35), ic(top)), (x1, ic(bottom)), shade, -1)
            cv2.rectangle(img, (x0, ic(top)), (x1, ic(bottom)), tuple(int(c * 0.55) for c in col), 1)
        else:
            u, v, d = data
            r = max(F_PX * BALL_R / d, 1.0)
            cv2.circle(img, (ic(u), ic(v)), int(round(r)), (35, 35, 205), -1, cv2.LINE_AA)
            cv2.circle(img, (ic(u - r * 0.3), ic(v - r * 0.3)), max(int(r * 0.35), 1),
                       (80, 80, 235), -1, cv2.LINE_AA)           # soft highlight

    # occlusion: a grey board passes in front of the ball
    for a, b in OCCLUSIONS:
        if a <= t <= b and tp:
            hw = int(max(F_PX * BALL_R / tp[2], 8) + 40)
            cv2.rectangle(img, (ic(tp[0]) - hw, 0), (ic(tp[0]) + hw, IMG_H), (70, 70, 75), -1)

    # lighting drift + sensor noise
    img = img.astype(np.float32) * (1.0 + 0.12 * math.sin(0.5 * t))
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), true_ball


# --------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def clean(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


def detect(img, min_area=12):
    """
    Two detectors on the same image.

    BALL     : red pixels -> contours -> must be ROUND (aspect ~1, fills its enclosing circle).
    OBSTACLE : every other saturated blob (other hues) + any red blob that failed the
               roundness test (e.g. the red barrel). Each blob -> (u_centre, v_bottom, w, h).

    Returns (ball (u,v,r) | None, obstacle_blobs, visualisation image).
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    red = clean(cv2.inRange(hsv, (0, 120, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 60), (180, 255, 255)))
    sat = clean(cv2.inRange(hsv, (11, 110, 50), (169, 255, 255)))     # saturated, non-red

    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball, ball_area, ball_cnt = None, 0, None
    red_other = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        _, _, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        fill = area / (math.pi * r * r + 1e-6)
        if 0.6 < aspect < 1.6 and fill > 0.6:
            if area > ball_area:
                if ball_cnt is not None:
                    red_other.append(ball_cnt)
                ball, ball_area, ball_cnt = (x, y, r), area, c
            else:
                red_other.append(c)
        else:
            red_other.append(c)                                    # red but NOT round -> obstacle

    cnts2, _ = cv2.findContours(sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in list(cnts2) + red_other:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < 40 or h < 8 or y + h <= CY + 3:
            continue
        blobs.append((x + w / 2.0, float(y + h), float(w), float(h), x <= 1, x + w >= IMG_W - 1))

    vis = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    vis[sat > 0] = (255, 0, 255)
    vis[red > 0] = (255, 255, 255)
    return ball, blobs, vis


class ObstacleMap:
    """
    Builds a tiny world map from camera blobs.

    Ground-plane geometry: an obstacle's bottom edge touches the floor, so its image row v
    gives depth  d = f * CAM_H / (v - cy).  Width in pixels gives its radius. Robot pose is
    assumed known from odometry (perfect in this simulator).
    """

    def __init__(self):
        self.items = []     # dicts: x, y, r, n

    def _nearest(self, wx, wy, gate):
        best, bd = None, 1e9
        for o in self.items:
            d = math.hypot(o["x"] - wx, o["y"] - wy)
            if d < bd:
                best, bd = o, d
        return (best, bd) if best is not None and bd < gate else (None, bd)

    def update(self, robot, blobs, ball=None):
        c, s = math.cos(robot.th), math.sin(robot.th)
        # everything that could hide part of an obstacle: other blobs and the ball
        occ = [(u - w / 2, u + w / 2, vb, vb - h) for u, vb, w, h, _, _ in blobs]
        if ball is not None:
            occ.append((ball[0] - ball[2], ball[0] + ball[2], ball[1] + ball[2], ball[1] - ball[2]))
        for i, (u, vb, w, h, cl, cr) in enumerate(blobs):
            l, rr = u - w / 2, u + w / 2
            hidden = False
            for j, (ol, orr, ob, ot) in enumerate(occ):
                if j == i or ob <= vb + 2 or ot > vb:     # skip itself, things not nearer, no vertical overlap
                    continue
                if ol <= l + 1 and orr >= rr - 1:
                    hidden = True                           # fully behind something nearer
                elif ol > l and orr < rr:
                    hidden = True                           # something nearer splits it in two
                elif ol <= rr + 3 and orr > rr:
                    cr = True                               # right side hidden by a nearer object
                elif orr >= l - 3 and ol < l:
                    cl = True                               # left side hidden by a nearer object
            dv = (vb - 0.5) - CY                            # bbox bottom sits ~0.5 px below the true edge
            if hidden or (cl and cr) or dv < 4:
                continue
            dn = F_PX * CAM_H / dv                          # depth of the near surface
            if dn > MAX_OBS_RANGE:
                continue
            k = min(w / (2 * F_PX), 0.7)
            r_est = float(np.clip(k * dn / (1 - k), 0.12, 0.6))
            partial = cl or cr                              # width unreliable -> locate from the visible edge

            def locate(r_use):
                dc = dn + r_use
                if not partial:
                    yl = (CX - u) * dc / F_PX
                elif cr:                                    # left edge visible
                    yl = (CX - l) * dc / F_PX - r_use
                else:                                       # right edge visible
                    yl = (CX - rr) * dc / F_PX + r_use
                return robot.x + dc * c - yl * s, robot.y + dc * s + yl * c

            r_use = 0.3 if partial else r_est
            wx, wy = locate(r_use)
            best, _ = self._nearest(wx, wy, 0.4 + 0.2 * dn)
            if best is not None and partial:
                r_use = best["r"]
                wx, wy = locate(r_use)
            if best is not None and partial and best["n"] > 20:
                best["n"] += 1                              # established obstacle: ignore unreliable partial views
            elif best is not None:
                a = 0.3 if dn < 2.0 else 0.1                # near measurements are more accurate
                best["x"] += a * (wx - best["x"])
                best["y"] += a * (wy - best["y"])
                if not partial:
                    best["r"] += a * (r_est - best["r"])
                best["n"] += 1
            else:
                self.items.append({"x": wx, "y": wy, "r": r_use, "n": 1})


def apply_avoidance(robot, omap, v_cmd, w_cmd, b_ball, state):
    """
    Free-space heading selection (a light-weight vector-field-histogram idea).

    Every 4 degrees in front of the robot we ask: "if I drove that way for LOOK metres, would
    I hit a mapped obstacle (inflated by robot radius + margin)?"  Each heading gets a cost =
    angle away from the ball + penalty if blocked + penalty for changing the last choice.
    If the straight line to the ball is free nothing changes (normal PID following). If it is
    blocked, the cheapest free heading is steered to and speed is scaled by how free it is.

    b_ball : bearing of the ball in the robot frame (rad, +left).
    Returns (v_cmd, w_cmd, active, closest_clearance).
    """
    c, s_ = math.cos(robot.th), math.sin(robot.th)
    obs, closest, block_rev = [], np.inf, False
    for o in omap.items:
        if o["n"] < MIN_SIGHTINGS:
            continue
        dx, dy = o["x"] - robot.x, o["y"] - robot.y
        xf, yl = c * dx + s_ * dy, -s_ * dx + c * dy
        rr = o["r"] + R_ROBOT + INFLATE
        closest = min(closest, math.hypot(xf, yl) - o["r"] - R_ROBOT)
        if xf < 0 and abs(yl) < rr and math.hypot(xf, yl) - rr < 0.4:
            block_rev = True
        if math.hypot(xf, yl) - rr < LOOK + 0.3:
            obs.append((xf, yl, rr))

    def free_dist(a):                                   # distance to first obstacle along heading a
        ca, sa = math.cos(a), math.sin(a)
        fd = LOOK
        for xf, yl, rr in obs:
            t = xf * ca + yl * sa
            if t <= 0:
                continue
            perp = abs(-xf * sa + yl * ca)
            if perp < rr:
                fd = min(fd, max(t - math.sqrt(rr * rr - perp * perp), 0.0))
        return fd

    if v_cmd < 0 and block_rev:
        v_cmd = 0.0
    if not obs:
        state["prev"] = None
        return v_cmd, w_cmd, False, closest
    ahead = float(np.clip((free_dist(0.0) - 0.1) / (LOOK - 0.1), 0.0, 1.0))   # room straight ahead
    if v_cmd > 0:
        v_cmd *= ahead                                  # the robot needs time to turn, so never rush an obstacle
    fd_ball = free_dist(b_ball)
    if fd_ball >= LOOK * 0.98:                          # straight line to the ball is clear
        state["prev"] = b_ball
        return v_cmd, w_cmd, ahead < 0.999, closest
    prev = state.get("prev")
    if prev is None:
        prev = b_ball
    best_a, best_cost, best_fd = None, 1e9, 0.0
    for a in np.radians(np.arange(-100, 101, 4)):
        fd = free_dist(a)
        cost = abs(a - b_ball) + W_BLOCK * (1.0 - fd / LOOK) + W_HYST * abs(a - prev)
        if cost < best_cost:
            best_a, best_cost, best_fd = a, cost, fd
    state["prev"] = best_a
    w_cmd = float(np.clip(K_H * best_a, -2.5, 2.5))
    if v_cmd > 0:
        v_cmd *= max(0.0, math.cos(best_a)) * float(np.clip((best_fd - 0.15) / (LOOK - 0.15), 0.0, 1.0))
        v_cmd = max(v_cmd, 0.0)
    return v_cmd, w_cmd, True, closest


# --------------------------------------------------------------------------
# Tracking: constant-velocity Kalman filter on (u, v, r)
# --------------------------------------------------------------------------
class KalmanTracker:
    """State = [u, v, r, du, dv, dr] (pixel position, ball radius, and their rates)."""

    def __init__(self, dt, accel_sigma=(350.0, 350.0, 60.0), meas_sigma=(2.0, 2.0, 1.5)):
        self.dt = dt
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        q = np.zeros((6, 6))
        for i, s in enumerate(accel_sigma):             # white-noise-acceleration model
            blk = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * s ** 2
            idx = [i, i + 3]
            q[np.ix_(idx, idx)] = blk
        self.Q = q
        self.R = np.diag(np.square(meas_sigma))
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e3
        self.initialised = False
        self.misses = 0

    def reset(self, z):
        self.x = np.array([z[0], z[1], z[2], 0, 0, 0], float)
        self.P = np.diag([4, 4, 2, 400, 400, 50.0])
        self.initialised = True
        self.misses = 0

    def predict(self, coasting=False):
        self.x = self.F @ self.x
        if coasting:                                    # don't trust old velocity while blind
            self.x[3:] *= 0.95
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Returns True if measurement accepted (passes Mahalanobis gate)."""
        z = np.asarray(z, float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > 30.0 and self.misses < 3:              # outlier -> reject
            self.misses += 1
            return False
        if d2 > 30.0:                                   # blind for a while -> re-acquire
            self.reset(z)
            return True
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.misses = 0
        return True


# --------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------
class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, i_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lo, self.hi, self.i_max = out_min, out_max, i_max
        self.i = 0.0
        self.prev = None

    def reset(self):
        self.i, self.prev = 0.0, None

    def __call__(self, err, dt, d_err=None):
        """d_err: externally supplied error derivative (e.g. from Kalman velocity)."""
        if d_err is None:
            d_err = 0.0 if self.prev is None else (err - self.prev) / dt
        self.prev = err
        self.i = float(np.clip(self.i + err * dt, -self.i_max, self.i_max))   # anti-windup
        out = self.kp * err + self.ki * self.i + self.kd * d_err
        return float(np.clip(out, self.lo, self.hi))


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
MAP_PX = 480
SC = MAP_PX / ARENA


def w2m(x, y):
    return int(x * SC), int(MAP_PX - y * SC)


def draw_map(robot, target, r_trail, t_trail, est_xy, omap):
    m = np.full((MAP_PX, MAP_PX, 3), 245, np.uint8)
    for g in range(0, int(ARENA) + 1):
        cv2.line(m, w2m(g, 0), w2m(g, ARENA), (225, 225, 225), 1)
        cv2.line(m, w2m(0, g), w2m(ARENA, g), (225, 225, 225), 1)
    for ox, oy, r, _, col in ALL_OBJECTS:
        cv2.circle(m, w2m(ox, oy), int(r * SC), col, -1, cv2.LINE_AA)
        cv2.circle(m, w2m(ox, oy), int(r * SC), tuple(int(c * 0.55) for c in col), 1, cv2.LINE_AA)
    for i in range(1, len(t_trail)):
        cv2.line(m, t_trail[i - 1], t_trail[i], (150, 150, 235), 1, cv2.LINE_AA)
    for i in range(1, len(r_trail)):
        cv2.line(m, r_trail[i - 1], r_trail[i], (200, 150, 70), 1, cv2.LINE_AA)
    ov = m.copy()                                              # field-of-view cone
    rc = w2m(robot.x, robot.y)
    half = math.radians(FOV_DEG / 2)
    pts = [rc] + [w2m(robot.x + 3.0 * math.cos(robot.th + a), robot.y + 3.0 * math.sin(robot.th + a))
                  for a in np.linspace(-half, half, 12)]
    cv2.fillPoly(ov, [np.array(pts, np.int32)], (230, 210, 150))
    m = cv2.addWeighted(ov, 0.45, m, 0.55, 0)
    for o in omap.items:                                       # vision-built obstacle map
        if o["n"] >= MIN_SIGHTINGS:
            cv2.circle(m, w2m(o["x"], o["y"]), max(int(o["r"] * SC), 3), (0, 150, 0), 1, cv2.LINE_AA)
    cv2.circle(m, rc, int(R_ROBOT * SC), (160, 90, 20), 1, cv2.LINE_AA)
    tri = [(robot.x + 0.28 * math.cos(robot.th), robot.y + 0.28 * math.sin(robot.th)),
           (robot.x + 0.2 * math.cos(robot.th + 2.5), robot.y + 0.2 * math.sin(robot.th + 2.5)),
           (robot.x + 0.2 * math.cos(robot.th - 2.5), robot.y + 0.2 * math.sin(robot.th - 2.5))]
    cv2.fillPoly(m, [np.array([w2m(*p) for p in tri], np.int32)], (160, 90, 20), cv2.LINE_AA)
    cv2.circle(m, w2m(target.x, target.y), 8, (40, 40, 215), -1, cv2.LINE_AA)
    if est_xy is not None:
        cv2.drawMarker(m, w2m(*est_xy), (0, 170, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    cv2.rectangle(m, (0, 0), (MAP_PX - 1, MAP_PX - 1), (120, 120, 120), 1)
    cv2.putText(m, "TOP-DOWN VIEW (ground truth)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(m, "red dot=target  green +=ball estimate  green ring=mapped obstacle", (8, MAP_PX - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    return m


def draw_camera(img, det, vis, blobs, kf_state, mode, info, avoiding):
    s = 2
    big = cv2.resize(img, (IMG_W * s, IMG_H * s), interpolation=cv2.INTER_LINEAR)
    cv2.line(big, (IMG_W * s // 2 - 12, IMG_H * s // 2), (IMG_W * s // 2 + 12, IMG_H * s // 2), (255, 255, 255), 1)
    cv2.line(big, (IMG_W * s // 2, IMG_H * s // 2 - 12), (IMG_W * s // 2, IMG_H * s // 2 + 12), (255, 255, 255), 1)
    for u, vb, w, h, _, _ in blobs:                           # obstacle boxes with ground-plane range
        dn = F_PX * CAM_H / max(vb - CY, 1.0)
        if dn <= MAX_OBS_RANGE:
            p0, p1 = (int((u - w / 2) * s), int((vb - h) * s)), (int((u + w / 2) * s), int(vb * s))
            cv2.rectangle(big, p0, p1, (255, 0, 255), 2)
            cv2.putText(big, f"OBSTACLE {dn:.1f}m", (p0[0], max(p0[1] - 5, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 0, 255), 1, cv2.LINE_AA)
    if det is not None:
        cv2.circle(big, (int(det[0] * s), int(det[1] * s)), int(det[2] * s) + 4, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(big, "BALL", (int(det[0] * s) - 16, int(det[1] * s) - int(det[2] * s) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    if kf_state is not None:
        col = (0, 215, 255) if mode == "TRACKING" else (0, 140, 255)
        cv2.drawMarker(big, (int(kf_state[0] * s), int(kf_state[1] * s)), col, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
    inset = cv2.resize(vis, (128, 96))
    h, w = inset.shape[:2]
    x0, y0 = big.shape[1] - w - 6, 60
    big[y0:y0 + h, x0:x0 + w] = inset
    cv2.rectangle(big, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    cv2.putText(big, "white=ball mask  magenta=other", (x0 - 96, y0 + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1, cv2.LINE_AA)
    colors = {"TRACKING": (0, 200, 0), "COASTING": (0, 140, 255), "SEARCHING": (0, 0, 230)}
    cv2.rectangle(big, (0, 0), (big.shape[1], 54), (30, 30, 30), -1)
    cv2.putText(big, "ROBOT CAMERA", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(big, mode, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[mode], 2, cv2.LINE_AA)
    if avoiding:
        cv2.putText(big, "AVOIDING", (190, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(big, info, (190, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return big


# --------------------------------------------------------------------------
# Main simulation loop
# --------------------------------------------------------------------------
def run(duration=40.0, use_kalman=True, seed=7, video_path=None, live=False, use_avoid=True):
    rng = np.random.default_rng(seed)
    robot, target = Robot(), Target()
    kf = KalmanTracker(DT)
    omap = ObstacleMap()
    avoid_state = {}
    ball_world = None
    pid_yaw = PID(2.6, 0.15, 0.35, -2.5, 2.5, i_max=0.5)
    pid_dist = PID(1.1, 0.05, 0.25, -0.8, 1.8, i_max=1.0)

    writer = None
    n = int(duration / DT)
    keys = ["t", "tx", "ty", "rx", "ry", "rth", "u_true", "r_true", "u_meas", "u_est", "bearing_true",
            "dist_true", "dist_est", "v_cmd", "w_cmd", "err_yaw", "clearance", "contact", "false_det", "avoiding"]
    log = {k: np.full(n, np.nan) for k in keys}
    log["mode"] = []
    r_trail, t_trail = [], []
    hist = []                                          # obstacle-map snapshots (for debugging/plots)
    last_sign = 1.0
    mode = "SEARCHING"

    for k in range(n):
        t = k * DT
        target.step(DT)
        img, true_ball = render_camera(robot, target, t, rng)
        det, blobs, vis = detect(img)
        omap.update(robot, blobs, det)
        if k % 15 == 0:
            hist.append((t, robot.x, robot.y, robot.th, [dict(o) for o in omap.items if o["n"] >= MIN_SIGHTINGS]))

        # ---------- ball state estimation ----------
        est = None
        got = False
        if use_kalman:
            if kf.initialised:
                kf.predict(coasting=(det is None or kf.misses > 0))
            if det is None:
                if kf.initialised:
                    kf.misses += 1
            elif not kf.initialised:
                kf.reset(det)
                got = True
            else:
                got = kf.update(det)
            if kf.initialised and kf.misses > MAX_COAST:
                kf.initialised = False                 # track lost
            if kf.initialised:
                est = kf.x.copy()
                mode = "TRACKING" if got else "COASTING"
            else:
                mode = "SEARCHING"
        else:                                           # baseline: raw detection, no memory
            if det is not None:
                est = np.array([det[0], det[1], det[2], 0, 0, 0], float)
                mode = "TRACKING"
            else:
                mode = "SEARCHING"

        # ---------- control ----------
        v_cmd = w_cmd = 0.0
        err_yaw = np.nan
        est_xy = None
        dist = np.nan
        avoiding = False
        clearance_seen = np.inf
        if est is not None:
            u, v_, r = est[0], est[1], max(est[2], 1.0)
            err_yaw = (CX - u) / F_PX                          # +ve => target is to the left
            dist = F_PX * BALL_R / r
            if use_kalman:
                d_err = -est[3] / F_PX                         # derivative from Kalman velocity
                dd = -F_PX * BALL_R * est[5] / (r * r)
            else:
                d_err = dd = None
            w_cmd = pid_yaw(err_yaw, DT, d_err)
            v_cmd = pid_dist(dist - D_TARGET, DT, dd)
            turn_first = max(0.0, 1.0 - abs(err_yaw) / 0.6)    # turn first, then drive
            last_sign = 1.0 if err_yaw >= 0 else -1.0
            bearing = math.atan2(CX - u, F_PX)
            est_xy = (robot.x + dist * math.cos(robot.th + bearing),
                      robot.y + dist * math.sin(robot.th + bearing))
            if got:
                ball_world = est_xy                            # remember where the ball is in the world
            if use_avoid:
                b_ball = bearing
                if mode == "COASTING" and ball_world is not None:   # view may have changed: use world memory
                    b_ball = math.atan2(ball_world[1] - robot.y, ball_world[0] - robot.x) - robot.th
                    b_ball = math.atan2(math.sin(b_ball), math.cos(b_ball))
                v_cmd, w_cmd, avoiding, clearance_seen = apply_avoidance(
                    robot, omap, v_cmd, w_cmd, b_ball, avoid_state)
            if not avoiding:                                   # while detouring, avoidance sets the speed itself
                v_cmd *= turn_first
        else:
            pid_yaw.reset()
            pid_dist.reset()
            w_cmd = 1.5 * last_sign                            # search: spin toward last seen side
            if ball_world is not None:                         # ...but first drive to where it was last seen
                dxl, dyl = ball_world[0] - robot.x, ball_world[1] - robot.y
                if math.hypot(dxl, dyl) > 1.0:
                    b_lk = math.atan2(dyl, dxl) - robot.th
                    b_lk = math.atan2(math.sin(b_lk), math.cos(b_lk))
                    w_cmd = float(np.clip(2.5 * b_lk, -2.5, 2.5))
                    v_cmd = 0.8 * max(0.0, math.cos(b_lk))
                    if use_avoid:
                        v_cmd, w_cmd, avoiding, clearance_seen = apply_avoidance(
                            robot, omap, v_cmd, w_cmd, b_lk, avoid_state)

        hit = robot.step(v_cmd, w_cmd, DT)

        # ---------- logging ----------
        xf, yl = to_robot_frame(robot, target.x, target.y)
        log["t"][k] = t
        log["tx"][k], log["ty"][k] = target.x, target.y
        log["rx"][k], log["ry"][k], log["rth"][k] = robot.x, robot.y, robot.th
        log["bearing_true"][k] = math.atan2(yl, xf) if xf > 0 else np.nan
        log["dist_true"][k] = math.hypot(target.x - robot.x, target.y - robot.y)
        if true_ball is not None:
            log["u_true"][k], log["r_true"][k] = true_ball[0], true_ball[2]
        if det is not None:
            log["u_meas"][k] = det[0]
            wrong = true_ball is None or math.hypot(det[0] - true_ball[0], det[1] - true_ball[1]) > max(10.0, 1.5 * true_ball[2])
            log["false_det"][k] = 1.0 if wrong else 0.0
        else:
            log["false_det"][k] = 0.0
        if est is not None:
            log["u_est"][k] = est[0]
            log["dist_est"][k] = dist
        log["clearance"][k] = min(math.hypot(robot.x - cx, robot.y - cy) - r - R_ROBOT for cx, cy, r in WORLD_CIRCLES)
        log["contact"][k] = 1.0 if hit else 0.0
        log["avoiding"][k] = 1.0 if avoiding else 0.0
        log["v_cmd"][k], log["w_cmd"][k], log["err_yaw"][k] = v_cmd, w_cmd, err_yaw
        log["mode"].append(mode)

        # ---------- rendering ----------
        if video_path or live:
            r_trail.append(w2m(robot.x, robot.y))
            t_trail.append(w2m(target.x, target.y))
            r_trail, t_trail = r_trail[-400:], t_trail[-400:]
            m = draw_map(robot, target, r_trail, t_trail, est_xy, omap)
            info = f"t={t:5.1f}s  dist={log['dist_true'][k]:.2f}m  v={robot.v:+.2f}  w={robot.w:+.2f}"
            cam = draw_camera(img, det, vis, blobs, est, mode, info, avoiding)
            frame = np.hstack([m, cam])
            if video_path and writer is None:
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         (frame.shape[1], frame.shape[0]))
            if writer is not None:
                writer.write(frame)
            if live:
                cv2.imshow("Vision-Based Tracking Simulator (q to quit)", frame)
                if cv2.waitKey(int(DT * 1000)) & 0xFF == ord("q"):
                    break
    if writer is not None:
        writer.release()
    if live:
        cv2.destroyAllWindows()
    log["mode"] = np.array(log["mode"])
    log["omap"] = [dict(o) for o in omap.items]
    log["omap_hist"] = hist
    return log


# --------------------------------------------------------------------------
# Metrics + plots
# --------------------------------------------------------------------------
def metrics(log, settle=5.0):
    t = log["t"]
    ok = ~np.isnan(t)
    after = (t > settle) & ok
    m = {}
    m["tracking_active_%"] = 100 * np.mean((log["mode"] != "SEARCHING")[after])
    b = np.degrees(np.abs(log["bearing_true"]))
    m["rms_bearing_err_deg"] = float(np.sqrt(np.nanmean(b[after] ** 2)))
    de = log["dist_true"] - D_TARGET
    m["rms_dist_err_m"] = float(np.sqrt(np.nanmean(de[after] ** 2)))
    m["contact_frames"] = int(np.nansum(log["contact"]))
    m["min_clearance_m"] = float(np.nanmin(log["clearance"]))
    m["false_ball_detections"] = int(np.nansum(log["false_det"]))
    items = [o for o in log["omap"] if o["n"] >= MIN_SIGHTINGS]
    found, errs = 0, []
    for x, y, r, _, _ in OBSTACLES:
        if items:
            d = [math.hypot(o["x"] - x, o["y"] - y) for o in items]
            j = int(np.argmin(d))
            if d[j] < 0.6:
                found += 1
                errs.append(d[j])
    m["obstacles_mapped_%"] = 100.0 * found / len(OBSTACLES)
    m["map_pos_err_m"] = float(np.mean(errs)) if errs else float("nan")
    m["time_avoiding_%"] = 100 * float(np.nanmean(log["avoiding"][after]))
    return m


def make_plots(A, B, C, path):
    """A = full system, B = no avoidance, C = no Kalman (raw detections)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_A, C_B, C_C, C_T, C_G = "#2a6fdb", "#d9822b", "#7a4fc1", "#c0392b", "#888888"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    t = A["t"]

    def shade(a):
        for s, e in OCCLUSIONS:
            a.axvspan(s, e, color=C_G, alpha=0.15, lw=0)

    a = ax[0, 0]
    for x, y, r, _, col in OBSTACLES:
        a.add_patch(plt.Circle((x, y), r, color=tuple(c / 255 for c in col[::-1]), alpha=0.9, zorder=3))
    a.plot(A["tx"], A["ty"], color=C_T, lw=1, label="Target path")
    a.plot(B["rx"], B["ry"], color=C_B, lw=1, ls="--", label="Robot, no avoidance")
    a.plot(A["rx"], A["ry"], color=C_A, lw=1.4, label="Robot, with avoidance")
    a.set_xlim(1, 8); a.set_ylim(1, 8.5); a.set_aspect("equal")
    a.set_title("Trajectories (top-down), coloured discs = obstacles")
    a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.legend(frameon=False, loc="upper right", fontsize=8)

    a = ax[0, 1]
    a.plot(t, A["clearance"], color=C_A, lw=1.3, label="With avoidance")
    a.plot(t, B["clearance"], color=C_B, lw=1.0, ls="--", label="No avoidance")
    a.axhline(0, color="k", lw=0.8)
    a.set_ylim(-0.05, 3.0)
    a.set_title("Clearance to nearest obstacle (0 = touching)")
    a.set_xlabel("time (s)"); a.set_ylabel("m"); a.legend(frameon=False, fontsize=8)

    a = ax[0, 2]
    a.plot(t, np.degrees(C["bearing_true"]), color=C_C, lw=1, label="No Kalman (raw detections)")
    a.plot(t, np.degrees(A["bearing_true"]), color=C_A, lw=1.2, label="Kalman + PID + avoidance")
    a.axhline(0, color="k", lw=0.6)
    a.axhline(35, color=C_G, lw=0.6, ls=":"); a.axhline(-35, color=C_G, lw=0.6, ls=":")
    shade(a); a.set_ylim(-90, 90)
    a.set_title("Target bearing from camera axis (dotted = FOV edge)")
    a.set_xlabel("time (s)"); a.set_ylabel("deg"); a.legend(frameon=False, loc="upper right", fontsize=8)

    a = ax[1, 0]
    a.plot(t, A["u_true"], color=C_T, lw=1.6, label="Ground truth")
    a.plot(t, A["u_meas"], ".", color=C_G, ms=2, label="Raw detection")
    a.plot(t, A["u_est"], color=C_A, lw=1.1, label="Kalman estimate")
    a.axhline(CX, color="k", lw=0.6, ls="--")
    shade(a); a.set_title("Ball horizontal position in image (grey = occlusion)")
    a.set_xlabel("time (s)"); a.set_ylabel("u (px)"); a.legend(frameon=False, ncol=3, loc="upper center", fontsize=8)
    a.set_ylim(-10, 330)

    a = ax[1, 1]
    a.plot(t, A["dist_true"], color=C_T, lw=1.4, label="True distance")
    a.plot(t, A["dist_est"], color=C_A, lw=1, label="Vision estimate")
    a.axhline(D_TARGET, color="k", lw=0.8, ls="--", label=f"Setpoint {D_TARGET} m")
    a.fill_between(t, 0, 6, where=A["avoiding"] > 0.5, color="#c040c0", alpha=0.15, lw=0, label="Avoiding")
    a.set_title("Following distance (purple = avoidance active)")
    a.set_xlabel("time (s)"); a.set_ylabel("m"); a.legend(frameon=False, fontsize=8, ncol=2); a.set_ylim(0, 6)

    a = ax[1, 2]
    a.axis("off")
    mA, mB, mC = metrics(A), metrics(B), metrics(C)
    rows = [("Contact frames with obstacles", "contact_frames", "{:d}"),
            ("Min clearance (m)", "min_clearance_m", "{:.2f}"),
            ("False ball detections", "false_ball_detections", "{:d}"),
            ("Obstacles mapped by vision (%)", "obstacles_mapped_%", "{:.0f}"),
            ("Obstacle position error (m)", "map_pos_err_m", "{:.2f}"),
            ("Tracking active (%)", "tracking_active_%", "{:.1f}"),
            ("RMS bearing error (deg)", "rms_bearing_err_deg", "{:.1f}"),
            ("RMS distance error (m)", "rms_dist_err_m", "{:.2f}"),
            ("Time avoiding (%)", "time_avoiding_%", "{:.1f}")]
    cells = [[lab, fmt.format(mA[k]), fmt.format(mB[k]), fmt.format(mC[k])] for lab, k, fmt in rows]
    tb = a.table(cellText=cells, colLabels=["Metric", "Full system", "No avoid", "No Kalman"],
                 loc="center", cellLoc="center", colWidths=[0.46, 0.18, 0.16, 0.18])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.55)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r_ == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#f0f0f0")
        if c_ == 0 and r_ > 0:
            cell._loc = "left"
    a.set_title("Metrics (after 5 s settle)")
    fig.suptitle("Vision-Based Tracking + Obstacle Avoidance - results", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return mA, mB, mC


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live", action="store_true", help="show a real-time window instead of writing a video")
    ap.add_argument("--no-avoid", action="store_true", help="disable obstacle avoidance (live mode)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    check_obstacles(a.duration)

    if a.live:
        run(a.duration, True, a.seed, live=True, use_avoid=not a.no_avoid)
        return
    raw_vid = os.path.join(a.out, "_raw.mp4")
    final_vid = os.path.join(a.out, "demo.mp4")
    print("Running full system (Kalman + PID + obstacle avoidance) ...")
    A = run(a.duration, True, a.seed, None if a.no_video else raw_vid, use_avoid=True)
    print("Running without avoidance ...")
    B = run(a.duration, True, a.seed, use_avoid=False)
    print("Running without Kalman filter ...")
    C = run(a.duration, False, a.seed, use_avoid=True)
    if not a.no_video:
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_vid, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "23", final_vid], check=True)
            os.remove(raw_vid)
        else:
            os.replace(raw_vid, final_vid)
        print("Video  ->", final_vid)
    mA, mB, mC = make_plots(A, B, C, os.path.join(a.out, "results.png"))
    print("Plots  ->", os.path.join(a.out, "results.png"))
    print(f"\n{'metric':26s}{'Full':>12s}{'No avoid':>12s}{'No Kalman':>12s}")
    for k in mA:
        print(f"{k:26s}{mA[k]:12.2f}{mB[k]:12.2f}{mC[k]:12.2f}")


if __name__ == "__main__":
    main()
