# Workspace: adaptive ROI + lane detection + steering overlay
# Paste your video path and any custom code below the marked blocks.

import cv2
import numpy as np
import math

# ==========================
# 1) Adaptive ROI detection
# ==========================

def detect_road_roi(image, prev_poly=None, beta=0.3):
    """
    Build an adaptive trapezoid ROI using a coarse horizon/vanishing-point estimate.
    Returns: mask (H×W uint8 0/255), poly (np.int32 [4,2])
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180, L2gradient=True)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    thetas = []
    if lines is not None:
        for rho_theta in lines[:200]:
            rho, theta = rho_theta[0]
            deg = np.degrees(theta)
            if 20 < deg < 80 or 100 < deg < 160:
                thetas.append(theta)

    def cluster_angles(ts, k=2):
        if len(ts) < 8:
            return None
        ts = np.array(ts, dtype=np.float32).reshape(-1,1)
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        _, _, centers = cv2.kmeans(ts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        return np.sort(centers.ravel())

    vp = None
    centers = cluster_angles(thetas, k=2) if thetas else None
    if centers is not None and lines is not None:
        def mean_line(theta_target, tol=np.deg2rad(7)):
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

        Ls = mean_line(centers[0])
        Rs = mean_line(centers[1])

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
            med = np.median(np.array(pts, dtype=np.float32), axis=0)
            vp = (float(med[0]), float(med[1]))

    if vp is not None:
        xv, yv = vp
        y_top = int(max(0.35*h, min(yv, 0.55*h)))   # higher top
        top_half_width = int(0.22*w)                # wider top
        bottom_margin = int(0.04*w)                 # tighter bottom
        poly = np.int32([
            [max(0, int(xv - top_half_width)), y_top],
            [min(w-1, int(xv + top_half_width)), y_top],
            [w - bottom_margin, h-1],
            [bottom_margin,     h-1],
        ])
    else:
        poly = np.int32([
            [int(0.45*w), int(0.45*h)],
            [int(0.55*w), int(0.45*h)],
            [int(0.96*w), h-1],
            [int(0.04*w), h-1],
        ])

    poly[:,0] = np.clip(poly[:,0], 0, w-1)
    poly[:,1] = np.clip(poly[:,1], 0, h-1)

    if prev_poly is not None:
        poly = (beta*poly + (1-beta)*prev_poly).astype(np.int32)  # temporal smoothing

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask, poly

# ==========================
# 2) Lane detection pieces
# ==========================

def roi_mask(img, poly):
    m = np.zeros_like(img)
    cv2.fillPoly(m, [poly], 255)
    return cv2.bitwise_and(img, m)

def detect_lines(frame, poly):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 180)
    masked = roi_mask(edges, poly)
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 30, minLineLength=40, maxLineGap=50)
    return lines, masked

def fit_lane(lines, w, h):
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
        z = np.polyfit(pts[:,1], pts[:,0], 2)  # x = a*y^2 + b*y + c
        return z
    return poly(left), poly(right)

# ==========================
# 3) Steering overlay
# ==========================

def steering_from_lanes(poly_L, poly_R, w, h, k1=40.0, k2=180.0, theta_max=25):
    y1, y2 = int(0.65*h), int(0.9*h)
    def x_from_poly(p,y): return p[0]*y*y + p[1]*y + p[2]
    if poly_L is None and poly_R is None:
        return 0.0
    if poly_L is None:
        lane_w = 0.45*w
        def midx(y): return x_from_poly(poly_R,y) - lane_w/2
    elif poly_R is None:
        lane_w = 0.45*w
        def midx(y): return x_from_poly(poly_L,y) + lane_w/2
    else:
        def midx(y):
            return 0.5*(x_from_poly(poly_L,y) + x_from_poly(poly_R,y))
    x1, x2 = midx(y1), midx(y2)
    xc = w/2
    e = (xc - x2) / w
    theta = math.atan2((x2 - x1), (y2 - y1))
    steering = k1*e + k2*theta*(180/math.pi)
    return max(-theta_max, min(theta_max, steering))


def overlay_wheel(frame, wheel_png, angle_deg, x=20, y=20, alpha=0.8, scale=0.6):
    h,w = frame.shape[:2]
    wh, ww = wheel_png.shape[:2]
    M = cv2.getRotationMatrix2D((ww/2, wh/2), angle_deg, scale)
    rot = cv2.warpAffine(wheel_png, M, (ww, wh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    bh, bw = rot.shape[:2]
    bh = min(bh, h-y); bw = min(bw, w-x)
    roi = frame[y:y+bh, x:x+bw]
    if rot.shape[2]==4:
        a = (rot[:bh,:bw,3:4]/255.0)*alpha
        rgb = rot[:bh,:bw,:3]
        roi[:] = (a*rgb + (1-a)*roi).astype(np.uint8)
    else:
        roi[:] = cv2.addWeighted(roi, 1-alpha, rot[:bh,:bw,:3], alpha, 0)
    return frame

# ==========================
# 4) Demo loop (plug your paths)
# ==========================
if __name__ == "__main__":
    cap = cv2.VideoCapture("VIDEO/20250330_084958_L.MP4")
    
    wheel = cv2.imread("resources/pngegg(2).png", cv2.IMREAD_UNCHANGED)  # optional overlay

    prev_poly = None
    steering_smoothed = 0.0
    alpha_ema = 0.2

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, poly = detect_road_roi(frame, prev_poly)
        prev_poly = poly.copy()
        lines, masked_edges = detect_lines(frame, poly)
        l,r = fit_lane(lines, frame.shape[1], frame.shape[0])
        steering = steering_from_lanes(l, r, frame.shape[1], frame.shape[0])
        steering_smoothed = alpha_ema*steering + (1-alpha_ema)*steering_smoothed

        vis = frame.copy()
        # draw ROI
        cv2.polylines(vis, [poly], True, (0,255,255), 2)
        # draw lanes
        def draw_lane(v, p, color=(0,255,0)):
            if p is None: return v
            h = v.shape[0]
            ys = np.linspace(int(0.6*h), h-1, 40)
            xs = p[0]*ys**2 + p[1]*ys + p[2]
            pts = np.int32(np.vstack([xs, ys]).T)
            for i in range(len(pts)-1):
                cv2.line(v, tuple(pts[i]), tuple(pts[i+1]), color, 6)
            return v
        vis = draw_lane(vis, l, (0,255,0))
        vis = draw_lane(vis, r, (0,255,0))

        if wheel is not None:
            vis = overlay_wheel(vis, wheel, -steering_smoothed)
        dir_txt = "RIGHT" if steering_smoothed > 3 else "LEFT" if steering_smoothed < -3 else "STRAIGHT"
        cv2.putText(vis, f"{dir_txt} ({steering_smoothed:.1f} deg)", (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("lane", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
