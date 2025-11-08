import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path)
    # Convert to grayscale if requested
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_video(video_path):
    """Load a video from a file path."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Video not found at {video_path}")
    return video


def create_roi_mask(image, vertices):
    """Create a single-channel binary mask for ROI.

    Returns a uint8 mask with shape (height, width) where inside-ROI=255 and outside=0.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def estimate_hood_cut_y(image, edge_thresh=0.02, smooth=15, min_search_frac=0.2):
    """Estimate a y-coordinate that separates the car hood/dashboard from the road.

    We compute Canny edges, sum edge intensity per row, smooth it and search from the
    bottom upward for the first row where normalized edge density exceeds `edge_thresh`.
    Returns a row index in [0, h-1]. If nothing is found, returns h-1.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # Normalize per row to [0..1] by dividing by max possible (255*w)
    row_sum = edges.sum(axis=1).astype(np.float32) / (255.0 * w + 1e-12)
    # Smooth the per-row signal to avoid single-row noise
    kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
    row_smooth = np.convolve(row_sum, kernel, mode='same')

    min_row = int(h * min_search_frac)
    # Search from bottom upwards for first row with enough edge density
    for y in range(h - 1, min_row - 1, -1):
        if row_smooth[y] > edge_thresh:
            return y
    return h - 1

def detect_road_roi(image):
    """
    Adaptive trapezoid ROI for lane detection.
    Returns: mask (uint8, 0/255), roi_poly (np.int32 of shape (4,2))
    """
    h, w = image.shape[:2]

    # 1) Light edges without prior ROI
    # image is RGB in this script, convert correctly to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180, L2gradient=True)

    # 2) Hough for coarse lane directions
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    thetas = []
    if lines is not None:
        for rho_theta in lines[:200]:  # limit work
            rho, theta = rho_theta[0]
            # keep non-horizontal and non-vertical extremes
            deg = np.degrees(theta)
            # typical lane lines are ~30..70° or 110..150° in image coords
            if 20 < deg < 80 or 100 < deg < 160:
                thetas.append(theta)

    # 3) Estimate vanishing point from two dominant angle clusters
    def cluster_angles(ts, k=2):
        if len(ts) < 8:
            return None
        ts = np.array(ts, dtype=np.float32).reshape(-1,1)
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        ret, labels, centers = cv2.kmeans(ts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        return np.sort(centers.ravel())

    vp = None
    centers = cluster_angles(thetas, k=2) if thetas else None
    if centers is not None and lines is not None:
        # Build two mean lines and intersect
        def mean_line(theta_target, tol=np.deg2rad(7)):
            pts = []
            for rho_theta in lines[:400]:
                rho, th = rho_theta[0]
                if abs(th - theta_target) <= tol:
                    # line in normal form: x cos th + y sin th = rho
                    # get two points on the line for intersection later
                    a, b = np.cos(th), np.sin(th)
                    x0, y0 = a*rho, b*rho
                    # points far apart on the line
                    p1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                    p2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                    pts.append((p1, p2))
            return pts

        Ls = mean_line(centers[0])
        Rs = mean_line(centers[1])

        def intersect(p1, p2, p3, p4):
            # line-line intersection
            x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
            den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if den == 0: return None
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/den
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/den
            return (px, py)

        inters = []
        for l1 in Ls[:10]:
            for l2 in Rs[:10]:
                p = intersect(l1[0], l1[1], l2[0], l2[1])
                if p is None: continue
                if -w < p[0] < 2*w and -h < p[1] < 2*h:
                    inters.append(p)
        if len(inters) >= 3:
            inters = np.array(inters, dtype=np.float32)
            # robust center of intersections
            med = np.median(inters, axis=0)
            vp = (float(med[0]), float(med[1]))

    # 4) Build trapezoid from vanishing point if available
    if vp is not None:
        xv, yv = vp
        # clamp vanishing y not to go below mid-height
        y_top = int(max(0.35*h, min(yv, 0.7*h)))
        # horizontal span grows with distance from vp
        top_half_width = int(0.10*w + 0.15*w * (1 - y_top/(h+1e-6)))
        bottom_margin = int(0.06*w)
        # estimate hood/dashboard boundary to exclude car interior
        bottom_y = estimate_hood_cut_y(image, edge_thresh=0.02, smooth=15, min_search_frac=0.2)
        poly = np.int32([
            [max(0, int(xv - top_half_width)),            y_top],
            [min(w-1, int(xv + top_half_width)),          y_top],
            [w - bottom_margin,                           bottom_y],
            [bottom_margin,                                bottom_y]
        ])
    else:
        # 5) Fallback: fixed trapezoid
        bottom_y = estimate_hood_cut_y(image, edge_thresh=0.02, smooth=15, min_search_frac=0.2)
        poly = np.int32([
            [int(0.45*w), int(0.60*h)],
            [int(0.55*w), int(0.60*h)],
            [int(0.92*w), bottom_y],
            [int(0.08*w), bottom_y]
        ])

    # 6) Smooth polygon corners slightly by keeping within image and convex hull
    poly[:,0] = np.clip(poly[:,0], 0, w-1)
    poly[:,1] = np.clip(poly[:,1], 0, h-1)
    hull = cv2.convexHull(poly.reshape(-1,1,2)).reshape(-1,2)
    if len(hull) >= 4:
        poly = hull

    # 7) Produce mask
    # ensure polygon has correct integer dtype for OpenCV
    poly = np.array(poly, dtype=np.int32)
    # reuse create_roi_mask to produce a single-channel binary mask
    mask = create_roi_mask(image, poly)
    return mask, poly

if __name__ == '__main__':
    # Example usage
    image_path = 'image/20250330_084958_L.MP4_snapshot_01.00.000.jpg'
    video_path = 'VIDEO/20250330_084958_L.MP4'

    try:
        # Load and process image
        image = load_image(image_path)
        image = cv2.resize(image, (960, 540))
        image = cv2.flip(image, 0)
        mask, polygon = detect_road_roi(image)
        # Apply mask to the RGB image (mask is single-channel uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Display the result
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Road ROI (masked image)')
        plt.imshow(masked_image)
        plt.axis('off')

        plt.show()

        # Load and process video (optional)
        video = load_video(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (960, 540))
            mask, polygon = detect_road_roi(frame_rgb)
            masked_frame = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
            # cv2.imshow expects BGR order
            bgr_masked = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Road ROI Video', bgr_masked)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
