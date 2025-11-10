# CLEAN VERSION — adaptive ROI, lane fit, smoothed steering, premultiplied RGBA overlay
# Drop-in script. Minimal globals. No hidden vars.

import cv2
import numpy as np
import math

# ==========================
# Config
# ==========================
TARGET_SIZE = (1280, 720)   # (W,H)
RESIZE_MODE  = "fit"        # "fit" letterbox | "stretch"

# Edge/Hough
CANNY_LOW, CANNY_HIGH = 60, 180
HOUGH_THRESH, HOUGH_MINLEN, HOUGH_MAXGAP = 30, 40, 50

# Lane drawing / smoothing
LANE_YMIN_RATIO = 0.55      # start of lane curves by height
EMA_POLY = 0.30             # EMA for polynomial coeffs
EMA_STEER = 0.15            # EMA for steering angle
RATE_LIMIT = 3.0            # deg per frame max change for the wheel
STEER_MAX = 25.0            # deg clamp

# ROI shaping (exclude sky/hood)
SKY_CROP = 0.35             # ignore top 35% for Hough/horizon
HOOD_CROP = 0.30            # ignore bottom 30% (car hood/dashboard)
ROI_TOP_HALF_WIDTH_RATIO = 0.30  # top half-width ratio of ROI
BOTTOM_MARGIN_X_RATIO   = 0.03   # left/right margin at bottom

# Wheel overlay + text
WHEEL_ANCHOR = "lb"         # 'lb','rb','lt','rt' (left/right + bottom/top)
WHEEL_OFFSET = (20, 20)     # (dx, dy) from anchor in px
WHEEL_SCALE  = 0.60         # visual scale for wheel
TEXT_BOTTOM_OFFSET = 28     # px from bottom for status text

# ==========================
# Resize helpers
# ==========================

def letterbox(frame, target_size):
    tw, th = target_size
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=frame.dtype)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def stretch(frame, target_size):
    tw, th = target_size
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

# ==========================
# ROI detection (adaptive with vanishing point, smoothed)
# ==========================

def detect_road_roi(image, prev_poly=None, beta=0.3):
    """Adaptive trapezoid ROI using coarse vanishing point; excludes sky and hood.
    Returns (mask, poly).
    """
    h, w = image.shape[:2]

    # Base edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, L2gradient=True)

    # Exclude sky and hood for Hough/horizon
    y_top_band = int(SKY_CROP * h)
    y_bot_band = int((1.0 - HOOD_CROP) * h)
    band = np.zeros_like(edges)
    band[y_top_band:y_bot_band, :] = 255
    edges_band = cv2.bitwise_and(edges, band)

    # Hough lines on the band only
    lines = cv2.HoughLines(edges_band, 1, np.pi/180, 120)
    thetas = []
    if lines is not None:
        for rho_theta in lines[:200]:
            _, theta = rho_theta[0]
            deg = np.degrees(theta)
            if 20 < deg < 80 or 100 < deg < 160:
                thetas.append(theta)

    def cluster_angles(ts, k=2):
        if len(ts) < 8: return None
        ts = np.array(ts, np.float32).reshape(-1,1)
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        _, _, centers = cv2.kmeans(ts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        return np.sort(centers.ravel())

    vp = None
    centers = cluster_angles(thetas, 2) if thetas else None
    if centers is not None and lines is not None:
        def collect(theta_target, tol=np.deg2rad(7)):
            segs = []
            for rho_theta in lines[:400]:
                rho, th = rho_theta[0]
                if abs(th - theta_target) <= tol:
                    a, b = np.cos(th), np.sin(th)
                    x0, y0 = a*rho, b*rho
                    p1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                    p2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                    segs.append((p1, p2))
            return segs
        Ls, Rs = collect(centers[0]), collect(centers[1])

        def intersect(p1, p2, p3, p4):
            x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
            den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if den == 0: return None
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/den
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/den
            return (px, py)

        pts = []
        for l1 in Ls[:10]:
            for l2 in Rs[:10]:
                p = intersect(l1[0], l1[1], l2[0], l2[1])
                if p is None: continue
                if -w < p[0] < 2*w and -h < p[1] < 2*h:
                    pts.append(p)
        if len(pts) >= 3:
            vp = tuple(np.median(np.array(pts, np.float32), axis=0))

    # Build trapezoid, clamped to the band (no sky, no hood)
    y_bottom = y_bot_band
    if vp is not None:
        xv, yv = vp
        y_top = int(np.clip(yv, y_top_band, int(0.60*h)))
        top_half = int(ROI_TOP_HALF_WIDTH_RATIO * w)
        bottom_margin = int(BOTTOM_MARGIN_X_RATIO * w)
        poly = np.int32([
            [max(0, int(xv - top_half)), y_top],
            [min(w-1, int(xv + top_half)), y_top],
            [w - bottom_margin, y_bottom],
            [bottom_margin,     y_bottom],
        ])
    else:
        bottom_margin = int(BOTTOM_MARGIN_X_RATIO * w)
        y_top = int(max(y_top_band, 0.45*h))
        top_half = int(ROI_TOP_HALF_WIDTH_RATIO * w)
        poly = np.int32([
            [int(0.5*w - top_half), y_top],
            [int(0.5*w + top_half), y_top],
            [w - bottom_margin,     y_bottom],
            [bottom_margin,         y_bottom],
        ])

    poly[:,0] = np.clip(poly[:,0], 0, w-1)
    poly[:,1] = np.clip(poly[:,1], 0, h-1)
    if prev_poly is not None:
        poly = (beta*poly + (1-beta)*prev_poly).astype(np.int32)

    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask, poly

# ==========================
# Lane detection and fitting
# ==========================

def roi_mask(img, poly):
    m = np.zeros_like(img)
    cv2.fillPoly(m, [poly], 255)
    return cv2.bitwise_and(img, m)


def detect_lines(frame, poly):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    masked = roi_mask(edges, poly)
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MINLEN, maxLineGap=HOUGH_MAXGAP)
    return lines, masked


def fit_lane(lines):
    if lines is None: return None, None
    left, right = [], []
    for x1,y1,x2,y2 in lines[:,0]:
        if x2==x1: continue
        k = (y2-y1)/(x2-x1)
        if abs(k) < 0.3: continue
        (left if k<0 else right).append(((x1+x2)/2, (y1+y2)/2))
    def poly(points):
        if len(points) < 3: return None
        pts = np.array(points)
        return np.polyfit(pts[:,1], pts[:,0], 2)
    return poly(left), poly(right)


def ema_poly(p_hat, p_prev, alpha=EMA_POLY):
    if p_hat is None: return p_prev
    if p_prev is None: return p_hat
    return alpha*p_hat + (1-alpha)*p_prev


def lane_points_from_poly(p, h, ymin_ratio=LANE_YMIN_RATIO, n=80):
    if p is None: return None
    ys = np.linspace(int(ymin_ratio*h), h-1, n)
    xs = p[0]*ys**2 + p[1]*ys + p[2]
    return np.int32(np.vstack([xs, ys]).T)

# ==========================
# Steering computation
# ==========================

def steering_from_lanes(poly_L, poly_R, w, h, k1=40.0, k2=180.0, theta_max=STEER_MAX):
    y1, y2 = int(0.65*h), int(0.9*h)
    def x_from_poly(p,y): return p[0]*y*y + p[1]*y + p[2]
    if poly_L is None and poly_R is None:
        return 0.0
    if poly_L is None:
        lane_w = 0.45*w
        midx = lambda y: x_from_poly(poly_R,y) - lane_w/2
    elif poly_R is None:
        lane_w = 0.45*w
        midx = lambda y: x_from_poly(poly_L,y) + lane_w/2
    else:
        midx = lambda y: 0.5*(x_from_poly(poly_L,y) + x_from_poly(poly_R,y))
    x1, x2 = midx(y1), midx(y2)
    xc = w/2
    e = (xc - x2) / w
    theta = math.atan2((x2 - x1), (y2 - y1))
    steering = k1*e + k2*theta*(180/math.pi)
    return max(-theta_max, min(theta_max, steering))

# ==========================
# Premultiplied RGBA overlay (no shape bugs)
# ==========================

def overlay_wheel(frame, wheel_rgba, angle_deg, x=20, y=20, alpha_mul=0.9, scale=0.6):
    h, w = frame.shape[:2]
    wh, ww = wheel_rgba.shape[:2]

    # premultiply RGBA
    rgba = wheel_rgba.astype(np.float32) / 255.0
    a = rgba[..., 3:4]                      # (H,W,1)
    rgb_p = rgba[..., :3] * a               # premultiplied

    # rotate both premultiplied RGB and alpha; transparent border
    M = cv2.getRotationMatrix2D((ww/2.0, wh/2.0), angle_deg, scale)
    rgb_rot = cv2.warpAffine(rgb_p, M, (ww, wh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a_rot   = cv2.warpAffine(a[...,0], M, (ww, wh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # (H,W)
    a_rot   = a_rot[..., None]               # -> (H,W,1)

    # target ROI on frame
    bh, bw = min(wh, h - y), min(ww, w - x)
    if bh <= 0 or bw <= 0:
        return frame
    roi = frame[y:y+bh, x:x+bw].astype(np.float32) / 255.0

    ar = (a_rot[:bh, :bw] * alpha_mul)      # (bh,bw,1)
    rgbp = (rgb_rot[:bh, :bw] * alpha_mul)  # (bh,bw,3)

    out = rgbp + roi*(1.0 - ar)             # composite in premul space
    frame[y:y+bh, x:x+bw] = (out*255.0).clip(0,255).astype(np.uint8)
    return frame

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cap = cv2.VideoCapture("VIDEO/20250330_115814_L.MP4")   # <— set your path
    wheel = cv2.imread("resources/wheel.png", cv2.IMREAD_UNCHANGED)  # optional

    prev_roi = None
    polyL, polyR = None, None
    angle_ema = 0.0
    angle_draw = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = letterbox(frame, TARGET_SIZE) if RESIZE_MODE=="fit" else stretch(frame, TARGET_SIZE)

        # ROI
        _, roi_poly = detect_road_roi(frame, prev_roi, beta=0.3)
        prev_roi = roi_poly.copy()

        # lines and fits
        lines, _ = detect_lines(frame, roi_poly)
        L_hat, R_hat = fit_lane(lines)
        polyL = ema_poly(L_hat, polyL)
        polyR = ema_poly(R_hat, polyR)

        # steering
        steer = steering_from_lanes(polyL, polyR, frame.shape[1], frame.shape[0])
        angle_ema = EMA_STEER*steer + (1-EMA_STEER)*angle_ema
        delta = np.clip(angle_ema - angle_draw, -RATE_LIMIT, RATE_LIMIT)
        angle_draw += delta

        # viz
        vis = frame.copy()
        cv2.polylines(vis, [roi_poly], True, (0,255,255), 2)
        def draw_curve(p, color=(0,255,0)):
            pts = lane_points_from_poly(p, vis.shape[0])
            if pts is not None:
                cv2.polylines(vis, [pts], False, color, 6)
        draw_curve(polyL)
        draw_curve(polyR)

        # wheel overlay with anchor and centered bottom text
        if wheel is not None:
            ww, wh = wheel.shape[1], wheel.shape[0]
            dx, dy = WHEEL_OFFSET
            anchor = WHEEL_ANCHOR.lower()
            x0 = dx if 'l' in anchor else vis.shape[1] - ww - dx
            y0 = dy if 't' in anchor else vis.shape[0] - wh - dy
            vis = overlay_wheel(vis, wheel, -angle_draw, x=x0, y=y0, scale=WHEEL_SCALE)
        # text centered at bottom
        status = f"{'RIGHT' if angle_draw>3 else 'LEFT' if angle_draw<-3 else 'STRAIGHT'} ({angle_draw:.1f} deg)"
        (font, fs, th) = (cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        (ts_w, ts_h), base = cv2.getTextSize(status, font, fs, th)
        cv2.putText(vis, status, ((vis.shape[1]-ts_w)//2, vis.shape[0]-TEXT_BOTTOM_OFFSET), font, fs, (0,255,0), th, cv2.LINE_AA)

        cv2.imshow("lane", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.release(); cv2.destroyAllWindows()
