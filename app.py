from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import base64

app = Flask(__name__)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
# --- Landmark IDs ---
NOSE            = 1
LEFT_TRAG       = 234
RIGHT_TRAG      = 454
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
LEFT_BASE       = 93      # jaw-contour anchor near left ear
LEFT_REF        = 123     # cheekbone ref to build outward direction
RIGHT_BASE      = 323
RIGHT_REF       = 352
FOREHEAD        = 10
CHIN            = 152

# --- Behaviour ---
VIS_RATIO_THRESH   = 0.75   # ear-to-nose / other-ear-to-nose ratio
EARLOBE_OFFSET     = 0.47   # push along jaw→ear direction to reach lobe
EARRING_SIZE_RATIO = 0.25   # earring height as fraction of face height
TILT_DAMPING       = 0.35   # 0 = earring ignores head roll, 1 = follows 100 %
FADE_SPEED         = 0.15   # opacity change per frame for show / hide
ANCHOR_Y_SHIFT     = 0.0    # fraction of earring height to shift attachment
                             # 0 = top edge, 0.1 = 10 % below top, etc.


# ╔══════════════════════════════════════════════════════════════════╗
# ║  ONE EURO FILTER  –  adaptive low-pass used by Snap / IG        ║
# ║                                                                  ║
# ║  • Low speed → heavy smoothing (kills jitter)                   ║
# ║  • High speed → light smoothing (kills lag)                     ║
# ║  Reference: Casiez et al., "1€ Filter", CHI 2012               ║
# ╚══════════════════════════════════════════════════════════════════╝
class OneEuroFilter:

    def __init__(self, t0, x0,
                 min_cutoff=1.0,   # Hz  – lower = smoother when still
                 beta=0.007,       # speed coefficient – higher = less lag when fast
                 d_cutoff=1.0):    # Hz  – derivative smoothing
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev     = float(x0)
        self.dx_prev    = 0.0
        self.t_prev     = t0

    # ── smoothing factor from cutoff frequency ──
    @staticmethod
    def _alpha(te, cutoff):
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def __call__(self, t, x):
        te = t - self.t_prev
        if te < 1e-9:
            te = 1e-6

        # smoothed derivative
        ad     = self._alpha(te, self.d_cutoff)
        dx     = (x - self.x_prev) / te
        dx_hat = ad * dx + (1.0 - ad) * self.dx_prev

        # adaptive cutoff: faster movement → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = self._alpha(te, cutoff)
        x_hat  = a * x + (1.0 - a) * self.x_prev

        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat


# ── convenience wrappers ──
class PointStabilizer:
    """Dual One-Euro filters for a 2-D point."""
    def __init__(self, **kw):
        self._fx = self._fy = None
        self._kw = kw

    def update(self, t, x, y):
        if self._fx is None:
            self._fx = OneEuroFilter(t, x, **self._kw)
            self._fy = OneEuroFilter(t, y, **self._kw)
            return x, y
        return self._fx(t, x), self._fy(t, y)


class ScalarStabilizer:
    """Single One-Euro filter for a scalar."""
    def __init__(self, **kw):
        self._f = None
        self._kw = kw

    def update(self, t, v):
        if self._f is None:
            self._f = OneEuroFilter(t, v, **self._kw)
            return v
        return self._f(t, v)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  AR EARRING TRACKER                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
class AREarTracker:

    def __init__(self, earring_path="earring.png"):

        # ── MediaPipe face mesh ──
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        # ── load earring BGRA ──
        img = cv2.imread(earring_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load '{earring_path}'")
        if img.shape[2] == 3:                       # add alpha if missing
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        self.earring_L = img                        # left side of image
        self.earring_R = cv2.flip(img, 1)           # mirrored for right side

        # ── per-ear stabilizers ──
        #  position  – responsive, low lag
        self.pos_L = PointStabilizer(min_cutoff=1.5, beta=0.5)
        self.pos_R = PointStabilizer(min_cutoff=1.5, beta=0.5)
        #  size      – heavier smoothing (size shouldn't jitter)
        self.sz_L  = ScalarStabilizer(min_cutoff=0.5, beta=0.05)
        self.sz_R  = ScalarStabilizer(min_cutoff=0.5, beta=0.05)

        # ── global stabilizers ──
        self.angle_stab  = ScalarStabilizer(min_cutoff=1.2, beta=0.3)
        self.faceh_stab  = ScalarStabilizer(min_cutoff=0.6, beta=0.05)

        # ── fade state ──
        self.opacity_L   = 0.0
        self.opacity_R   = 0.0
        self.cache_L     = None      # (x, y, sz)
        self.cache_R     = None
        self.cache_angle = 0.0

        self.t0 = time.time()

    # ─────────────────── helpers ───────────────────
    def _t(self):
        return time.time() - self.t0

    def _visibility(self, lm, w, h):
        """Which ears are geometrically visible (not occluded by head turn)."""
        nose = np.array([lm[NOSE].x * w, lm[NOSE].y * h])
        L    = np.array([lm[LEFT_TRAG].x * w,  lm[LEFT_TRAG].y * h])
        R    = np.array([lm[RIGHT_TRAG].x * w, lm[RIGHT_TRAG].y * h])
        ld, rd = np.linalg.norm(L - nose), np.linalg.norm(R - nose)
        return (ld / (rd + 1e-6) > VIS_RATIO_THRESH,
                rd / (ld + 1e-6) > VIS_RATIO_THRESH)

    def _head_roll_deg(self, lm, w, h):
        """Roll angle from outer eye corners (degrees)."""
        le = np.array([lm[LEFT_EYE_OUTER].x * w,  lm[LEFT_EYE_OUTER].y * h])
        re = np.array([lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_OUTER].y * h])
        return math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))

    def _face_height(self, lm, w, h):
        """Euclidean distance forehead → chin (pixels)."""
        f = np.array([lm[FOREHEAD].x * w, lm[FOREHEAD].y * h])
        c = np.array([lm[CHIN].x * w,     lm[CHIN].y * h])
        return np.linalg.norm(c - f)

    def _yaw_factors(self, lm, w, h):
        """
        Per-ear scale multiplier driven by face yaw.
        Frontal → 1.0 ;  turned away → shrinks to 0.4
        Gives a natural perspective / foreshortening effect.
        """
        nose = np.array([lm[NOSE].x * w, lm[NOSE].y * h])
        L    = np.array([lm[LEFT_TRAG].x * w,  lm[LEFT_TRAG].y * h])
        R    = np.array([lm[RIGHT_TRAG].x * w, lm[RIGHT_TRAG].y * h])
        ld, rd = np.linalg.norm(L - nose), np.linalg.norm(R - nose)
        s = ld + rd + 1e-6
        return (np.clip(ld / s * 2.0, 0.4, 1.0),
                np.clip(rd / s * 2.0, 0.4, 1.0))

    def _estimate_lobe(self, lm, base_id, ref_id, w, h):
        """
        Project from a jaw-contour landmark outward along the
        jaw → ear direction to estimate the earlobe position.
        """
        bx, by = lm[base_id].x, lm[base_id].y
        dx = bx - lm[ref_id].x
        dy = by - lm[ref_id].y
        px = bx + dx * EARLOBE_OFFSET
        py = by + dy * EARLOBE_OFFSET
        return px * w, py * h

    # ─────────────────── overlay ───────────────────
    def _overlay(self, frame, earring_src, cx, cy, size, angle_deg, opacity):
        """
        Composite a BGRA earring onto the frame with:
          • correct sizing
          • rotation about the attachment point (top-centre + ANCHOR_Y_SHIFT)
          • per-pixel alpha blending scaled by current opacity
        """
        if size < 4 or opacity < 0.01:
            return
        fh, fw = frame.shape[:2]

        # ── resize keeping aspect ratio ──
        aspect = earring_src.shape[1] / earring_src.shape[0]
        nh = max(int(size), 2)
        nw = max(int(nh * aspect), 2)
        ear = cv2.resize(earring_src, (nw, nh), interpolation=cv2.INTER_AREA)

        # ── rotate around attachment point ──
        #    attachment = top-centre shifted down by ANCHOR_Y_SHIFT
        att_x = nw // 2
        att_y = int(nh * ANCHOR_Y_SHIFT)
        angle_applied = angle_deg * TILT_DAMPING   # gravity damping
        M = cv2.getRotationMatrix2D((att_x, att_y), -angle_applied, 1.0)

        # expanded canvas so nothing is cropped
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        rw = int(nh * sin_a + nw * cos_a) + 2
        rh = int(nh * cos_a + nw * sin_a) + 2
        M[0, 2] += (rw - nw) / 2
        M[1, 2] += (rh - nh) / 2

        ear = cv2.warpAffine(ear, M, (rw, rh),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

        # where did the attachment point land after rotation?
        new_att = M @ np.array([att_x, att_y, 1.0])
        ox = int(cx - new_att[0])
        oy = int(cy - new_att[1])

        # ── clip to frame bounds ──
        eh, ew = ear.shape[:2]
        x1, y1, x2, y2       = ox, oy, ox + ew, oy + eh
        sx1, sy1, sx2, sy2   = 0,  0,  ew,      eh
        if x1 < 0:   sx1 -= x1;  x1 = 0
        if y1 < 0:   sy1 -= y1;  y1 = 0
        if x2 > fw:  sx2 -= (x2 - fw);  x2 = fw
        if y2 > fh:  sy2 -= (y2 - fh);  y2 = fh
        if x1 >= x2 or y1 >= y2:
            return

        crop = ear[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return

        # ── alpha-blend ──
        alpha = (crop[:, :, 3:4].astype(np.float32) / 255.0) * opacity
        fg    = crop[:, :, :3].astype(np.float32)
        roi   = frame[y1:y2, x1:x2].astype(np.float32)
        frame[y1:y2, x1:x2] = (fg * alpha + roi * (1.0 - alpha)).astype(np.uint8)

    # ─────────────────── per-frame entry point ───────────────────
    def process(self, frame):
        h, w = frame.shape[:2]
        t    = self._t()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        # ── no face: fade both earrings out gracefully ──
        if not results.multi_face_landmarks:
            self.opacity_L = max(0.0, self.opacity_L - FADE_SPEED)
            self.opacity_R = max(0.0, self.opacity_R - FADE_SPEED)
            if self.cache_L and self.opacity_L > 0.01:
                self._overlay(frame, self.earring_L,
                              *self.cache_L, self.cache_angle, self.opacity_L)
            if self.cache_R and self.opacity_R > 0.01:
                self._overlay(frame, self.earring_R,
                              *self.cache_R, self.cache_angle, self.opacity_R)
            return frame

        lm = results.multi_face_landmarks[0].landmark
        show_L, show_R = self._visibility(lm, w, h)

        # ── face-wide metrics (stabilised) ──
        fh       = self.faceh_stab.update(t, self._face_height(lm, w, h))
        angle    = self.angle_stab.update(t, self._head_roll_deg(lm, w, h))
        lf, rf   = self._yaw_factors(lm, w, h)
        base_sz  = fh * EARRING_SIZE_RATIO

        # ── LEFT earring ──
        if show_L:
            x, y = self._estimate_lobe(lm, LEFT_BASE, LEFT_REF, w, h)
            x, y = self.pos_L.update(t, x, y)
            sz   = self.sz_L.update(t, base_sz * lf)
            self.opacity_L = min(1.0, self.opacity_L + FADE_SPEED)
            self.cache_L   = (x, y, sz)
        else:
            self.opacity_L = max(0.0, self.opacity_L - FADE_SPEED)

        if self.cache_L and self.opacity_L > 0.01:
            self._overlay(frame, self.earring_L,
                          *self.cache_L, angle, self.opacity_L)

        # ── RIGHT earring ──
        if show_R:
            x, y = self._estimate_lobe(lm, RIGHT_BASE, RIGHT_REF, w, h)
            x, y = self.pos_R.update(t, x, y)
            sz   = self.sz_R.update(t, base_sz * rf)
            self.opacity_R = min(1.0, self.opacity_R + FADE_SPEED)
            self.cache_R   = (x, y, sz)
        else:
            self.opacity_R = max(0.0, self.opacity_R - FADE_SPEED)

        if self.cache_R and self.opacity_R > 0.01:
            self._overlay(frame, self.earring_R,
                          *self.cache_R, angle, self.opacity_R)

        self.cache_angle = angle
        return frame


# ╔══════════════════════════════════════════════════════════════════╗
# ║  FLASK WEB APP                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

tracker = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    from flask import request, jsonify
    global tracker
    
    if tracker is None:
        tracker = AREarTracker("earring.png")
    
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        processed = tracker.process(frame)
        
        _, buffer = cv2.imencode('.jpg', processed)
        img_str = base64.b64encode(buffer).decode()
        
        return jsonify({'image': f'data:image/jpeg;base64,{img_str}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)