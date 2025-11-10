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
EMA_STEER = 0.05            # EMA for steering angle — ОЧЕНЬ ПЛАВНОЕ (было 0.15)
RATE_LIMIT = 1.5            # deg per frame max change for the wheel — СТРОГОЕ (было 3.0)
STEER_MAX = 25.0            # deg clamp

# Lane memory / interpolation
LANE_MEMORY_FRAMES = 40      # уменьшено с 120 → 40, чтобы избежать залипания на смене полос
LANE_FADE_ALPHA = 0.3       # полупрозрачность для "памяти" (0.0-1.0)
LANE_POLY_DECAY = 0.98      # коэффициент деградации полинома без детекции (0.98 = -2% каждый кадр)

# ROI shaping (exclude sky/hood)
SKY_CROP = 0.35             # ignore top 35% for Hough/horizon
HOOD_CROP = 0.315            # ignore bottom 33% (car hood/dashboard)
ROI_TOP_HALF_WIDTH_RATIO = 0.30  # top half-width ratio of ROI
BOTTOM_MARGIN_X_RATIO   = 0.03   # left/right margin at bottom

# Wheel overlay + text
WHEEL_ANCHOR = "lb"         # 'lb','rb','lt','rt' (left/right + bottom/top)
WHEEL_OFFSET = (20, 20)     # (dx, dy) from anchor in px
WHEEL_SCALE  = 0.60         # visual scale for wheel
TEXT_BOTTOM_OFFSET = 28     # px from bottom for status text

# ==========================
# Image processing helpers
# ==========================

def compute_edges(frame):
    """Compute Canny edges from frame.

    Вычисляет edges один раз, чтобы избежать повторения логики.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, L2gradient=True)
    return edges


def create_y_mask(h, w, y_min, y_max):
    """Create Y-coordinate mask (band).

    Создаёт маску для ограничения области по Y координатам.
    """
    mask = np.zeros((h, w), np.uint8)
    mask[y_min:y_max, :] = 255
    return mask


def apply_morphology(edges, kernel_size=(5, 5), iterations_close=2, iterations_erode=1):
    """Apply morphological operations to detect dashed lines.

    Заполняет разрывы в штрих-пунктирных линиях.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)

    kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_processed = cv2.morphologyEx(edges_closed, cv2.MORPH_ERODE, kernel_thin, iterations=iterations_erode)

    return edges_processed


def line_y_at_center(rho, theta, x_center):
    """Вычислить Y координату линии в центре кадра.

    Args:
        rho, theta: Параметры линии из Hough
        x_center: X координата центра кадра

    Returns:
        Y координата или None если деление на 0
    """
    a = np.cos(theta)
    b = np.sin(theta)
    if abs(b) < 0.01:
        return None
    return (rho - a * x_center) / b


def line_intersection(p1, p2, p3, p4):
    """Найти пересечение двух линий.

    Args:
        p1, p2: Точки первой линии
        p3, p4: Точки второй линии

    Returns:
        (x, y) пересечения или None если параллельны
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den

    return (px, py)


# ==========================
# Geometry helpers
# ==========================

def cluster_angles(angles, k=2):
    """Кластеризация углов линий через k-means.

    Args:
        angles: Список углов (радианы)
        k: Количество кластеров

    Returns:
        Отсортированные центры кластеров или None
    """
    if len(angles) < 8:
        return None

    ts = np.array(angles, np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    _, _, centers = cv2.kmeans(ts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    return np.sort(centers.ravel())


def find_vanishing_point_ransac(pts, weights=None, iterations=100, threshold=30.0, min_inliers=3):
    """Найти vanishing point используя RANSAC.

    RANSAC более надёжен чем медиана на развязках/бордюрах, так как:
    1. Итеративно проверяет гипотезы
    2. Отбрасывает выбросы (spurious intersections)
    3. Находит consensual solution

    Args:
        pts: Список точек пересечений
        weights: Опциональные веса (длины отрезков)
        iterations: Количество RANSAC итераций
        threshold: Максимальное расстояние до модели (пиксели)
        min_inliers: Минимум inliers для валидной модели

    Returns:
        (x, y) vanishing point или None
    """
    if len(pts) < min_inliers:
        return None

    pts_arr = np.array(pts, np.float32)

    best_vp = None
    best_inliers_count = 0

    for iteration in range(iterations):
        # Случайно выбираем одну точку как гипотезу VP
        idx = np.random.randint(0, len(pts_arr))
        vp_candidate = pts_arr[idx]

        # Считаем сколько других точек близко к этой (inliers)
        distances = np.linalg.norm(pts_arr - vp_candidate, axis=1)
        inliers_mask = distances < threshold
        inliers_count = np.sum(inliers_mask)

        # Если это лучшая гипотеза, сохраняем её
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            # Уточняем VP как среднее inliers (взвешенное если есть weights)
            if inliers_count >= min_inliers:
                inlier_pts = pts_arr[inliers_mask]
                if weights is not None:
                    inlier_weights = np.array(weights)[inliers_mask]
                    inlier_weights = inlier_weights / np.sum(inlier_weights)
                    best_vp = np.average(inlier_pts, axis=0, weights=inlier_weights)
                else:
                    best_vp = np.mean(inlier_pts, axis=0)

    return tuple(best_vp) if best_vp is not None else None


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

def detect_road_roi(image, edges, prev_poly=None, beta=0.3):
    """Adaptive trapezoid ROI using coarse vanishing point; excludes sky and hood.

    Args:
        image: Input frame
        edges: Pre-computed Canny edges (не вычисляем заново)
        prev_poly: Previous polygon for smoothing
        beta: EMA coefficient for polygon smoothing
    """
    h, w = image.shape[:2]

    # Исключаем верхнюю (небо) и нижнюю (капот) части для определения горизонта
    y_top_band = int(SKY_CROP * h)
    y_bot_band = int((1.0 - HOOD_CROP) * h)
    band = create_y_mask(h, w, y_top_band, y_bot_band)
    edges_band = cv2.bitwise_and(edges, band)

    # Hough lines — ищем линии горизонта для vanishing point
    lines = cv2.HoughLines(edges_band, 1, np.pi/180, 120)
    thetas = []
    if lines is not None:
        for rho_theta in lines[:200]:
            _, theta = rho_theta[0]
            deg = np.degrees(theta)
            if 20 < deg < 80 or 100 < deg < 160:
                thetas.append(theta)

    # Автоматическое определение нижней границы (горизонта капота)
    # УЯЗВИМОСТЬ FIX: ограничиваем поиск в последних 8-10% кадра, чтобы не ловить HUD/дешборд
    y_hood_search_start = int((1.0 - HOOD_CROP) * h)
    y_hood_search_end = int(0.95 * h)  # только последние 5% вместо всего
    hood_band = create_y_mask(h, w, y_hood_search_start, y_hood_search_end)
    edges_hood = cv2.bitwise_and(edges, hood_band)

    hood_lines = cv2.HoughLines(edges_hood, 1, np.pi/180, 100)
    hood_ys = []
    if hood_lines is not None:
        for rho_theta in hood_lines[:100]:
            rho, theta = rho_theta[0]
            deg = np.degrees(theta)
            if deg < 15 or deg > 165:
                y = line_y_at_center(rho, theta, w / 2)
                if y is not None and y_hood_search_start <= y <= y_hood_search_end:
                    hood_ys.append(y)

    y_hood = int(np.median(hood_ys)) if len(hood_ys) >= 2 else y_bot_band

    # Найти vanishing point
    vp = None
    centers = cluster_angles(thetas, 2) if thetas else None
    if centers is not None and lines is not None:
        # Собрать сегменты линий по углам
        def collect_line_segments(theta_target, tol=np.deg2rad(7)):
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

        Ls = collect_line_segments(centers[0])
        Rs = collect_line_segments(centers[1])

        # УЯЗВИМОСТЬ FIX: используем RANSAC для поиска vanishing point (вместо взвешенной медианы)
        # RANSAC более устойчив к развязкам/бордюрам
        pts = []
        weights = []
        for l1 in Ls[:10]:
            for l2 in Rs[:10]:
                p = line_intersection(l1[0], l1[1], l2[0], l2[1])
                if p is not None and -w < p[0] < 2*w and -h < p[1] < 2*h:
                    pts.append(p)
                    # Вес = минимальная длина отрезка (более длинные отрезки надёжнее)
                    len1 = np.sqrt((l1[1][0]-l1[0][0])**2 + (l1[1][1]-l1[0][1])**2)
                    len2 = np.sqrt((l2[1][0]-l2[0][0])**2 + (l2[1][1]-l2[0][1])**2)
                    weights.append(min(len1, len2))

        if len(pts) >= 3:
            # RANSAC находит точку с максимальным консенсусом (inliers)
            vp = find_vanishing_point_ransac(pts, weights=weights, iterations=100, threshold=40.0)

    # Построить trapezoid ROI
    y_bottom = y_hood
    top_half = int(ROI_TOP_HALF_WIDTH_RATIO * w)
    bottom_margin = int(BOTTOM_MARGIN_X_RATIO * w)

    if vp is not None:
        xv, yv = vp
        y_top = int(np.clip(yv, y_top_band, int(0.60*h)))
        x_left = int(xv - top_half)
        x_right = int(xv + top_half)
    else:
        y_top = int(max(y_top_band, 0.45*h))
        x_left = int(0.5*w - top_half)
        x_right = int(0.5*w + top_half)

    poly = np.int32([
        [max(0, x_left), y_top],
        [min(w-1, x_right), y_top],
        [w - bottom_margin, y_bottom],
        [bottom_margin, y_bottom],
    ])

    poly[:,0] = np.clip(poly[:,0], 0, w-1)
    poly[:,1] = np.clip(poly[:,1], 0, h-1)

    if prev_poly is not None:
        poly = (beta*poly + (1-beta)*prev_poly).astype(np.int32)

    return poly

# ==========================
# Lane tracking with memory
# ==========================

class LaneTracker:
    """Отслеживание линии с механизмом памяти и деградацией.

    Если линия не обнаружена в текущем фрейме, используется предыдущая
    в течение LANE_MEMORY_FRAMES кадров с деградацией коэффициентов.

    УЯЗВИМОСТЬ FIX:
    - Хранит lane_width из предыдущих кадров (для адаптивной ширины полос)
    - Деградирует полином каждый кадр без детекции (умножает на 0.98)
    - Это предотвращает "залипание" на смене полос/поворотах
    """
    def __init__(self):
        self.poly = None           # Полином текущей линии
        self.frames_missing = 0    # Счётчик кадров без обнаружения
        self.lane_width = None     # Ширина полосы из предыдущих кадров

    def update(self, p_new):
        """Обновить состояние линии.

        Args:
            p_new: Новый обнаруженный полином (может быть None)

        Returns:
            poly: Полином для отрисовки (с учётом памяти)
            is_detected: True если линия обнаружена в этом фрейме
            is_memory: True если рисуем "по памяти"
        """
        is_detected = p_new is not None
        is_memory = False

        if is_detected:
            # Линия обнаружена — обновляем полином и обнуляем счётчик
            self.poly = ema_poly(p_new, self.poly)
            self.frames_missing = 0
        else:
            # Линия не обнаружена
            self.frames_missing += 1
            is_memory = self.frames_missing <= LANE_MEMORY_FRAMES

            # УЯЗВИМОСТЬ FIX: деградируем полином без детекции (умножаем на 0.98)
            # Это предотвращает "залипание" — коэффициенты плавно идут к 0
            if self.poly is not None and self.frames_missing <= LANE_MEMORY_FRAMES:
                self.poly = self.poly * LANE_POLY_DECAY

            # Если превышен лимит памяти, забываем линию
            if self.frames_missing > LANE_MEMORY_FRAMES:
                self.poly = None

        return self.poly, is_detected, is_memory

    def get_lane_width(self):
        """Получить адаптивную ширину полосы из истории."""
        return self.lane_width

    def set_lane_width(self, width):
        """Сохранить ширину полосы для использования при пропаже одной стороны."""
        if width is not None and width > 0:
            self.lane_width = width

# ==========================


def detect_lines(frame, edges, poly, ymin=None, ymax=None):
    """Detect lane lines within ROI bounds.

    Args:
        frame: Input frame (для размеров)
        edges: Pre-computed Canny edges (не вычисляем заново)
        poly: ROI polygon
        ymin: Minimum Y coordinate (top of ROI)
        ymax: Maximum Y coordinate (bottom of ROI)
    """
    h, w = frame.shape[:2]

    if ymin is None:
        ymin = int(np.min(poly[:, 1]))
    if ymax is None:
        ymax = int(np.max(poly[:, 1]))

    # УЯЗВИМОСТЬ FIX: исключаем 2-3 px рамку по краям letterbox
    # Чтобы Hough не ловил резкую вертикальную границу
    edges_trimmed = edges.copy()
    edges_trimmed[:3, :] = 0
    edges_trimmed[-3:, :] = 0
    edges_trimmed[:, :3] = 0
    edges_trimmed[:, -3:] = 0

    # УЯЗВИМОСТЬ FIX: анизотропная морфология для скоростной трассы
    # Используем прямоугольник 15x3 (длинный горизонтально) вместо эллипса 5x5
    # Это лучше для полос на скоростной трассе и не "распухает" края
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    edges_processed = cv2.morphologyEx(edges_trimmed, cv2.MORPH_CLOSE, kernel_rect, iterations=1)

    # Создаём две маски
    roi_mask_poly = np.zeros_like(edges_processed)
    cv2.fillPoly(roi_mask_poly, [poly], 255)

    y_bound_mask = create_y_mask(h, w, ymin, ymax)

    # Объединяем маски
    edges_bounded = cv2.bitwise_and(edges_processed, cv2.bitwise_and(roi_mask_poly, y_bound_mask))

    # Поиск линий в ограниченной области
    lines = cv2.HoughLinesP(edges_bounded, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MINLEN, maxLineGap=HOUGH_MAXGAP)
    return lines


def fit_lane(lines):
    """Fit polynomial to left and right lane lines with validation.

    Отфильтровывает шумовые точки и валидирует данные перед подгонкой.
    ФИЛЬТРАЦИЯ ПО УГЛУ: исключаем почти горизонтальные линии, которые приводят к прыжкам.
    """
    if lines is None: return None, None
    left, right = [], []
    for x1,y1,x2,y2 in lines[:,0]:
        if x2==x1: continue
        k = (y2-y1)/(x2-x1)
        # СТРОГАЯ ФИЛЬТРАЦИЯ: угол должен быть довольно крутым (|k| > 0.5)
        # Это исключает горизонтальные линии, которые вызывают прыжки
        # машина не может ездить горизонтально, полосы всегда под углом
        if abs(k) < 0.5: continue
        if abs(k) > 3.0: continue  # исключаем очень вертикальные линии (шум)
        (left if k<0 else right).append(((x1+x2)/2, (y1+y2)/2))

    def poly(points):
        """Подгонка полинома с валидацией данных."""
        if len(points) < 3:
            return None

        pts = np.array(points, dtype=np.float32)

        # ФИЛЬТРАЦИЯ: убираем явные выбросы (outliers)
        # Считаем стандартное отклонение по X и отфильтровываем точки > 2 сигмы
        x_vals = pts[:, 0]
        x_mean = np.mean(x_vals)
        x_std = np.std(x_vals)

        if x_std > 0:
            # Оставляем только точки в пределах 2.5 сигм от среднего
            valid_mask = np.abs(x_vals - x_mean) <= 2.5 * x_std
            pts = pts[valid_mask]

        # Убедиться, что после фильтрации осталось достаточно точек
        if len(pts) < 3:
            return None

        # Проверяем коллинеарность: если все точки почти на одной линии (Y не меняется)
        y_vals = pts[:, 1]
        y_std = np.std(y_vals)
        if y_std < 1e-6:  # Y координаты почти не меняются → плохие данные
            return None

        # Подгонка с подавлением warning'а для плохо обусловленных систем
        with np.errstate(all='ignore'):
            try:
                result = np.polyfit(pts[:, 1], pts[:, 0], 2)
                # Проверяем, что коэффициенты разумные (не NaN/Inf)
                if np.any(~np.isfinite(result)):
                    return None
                return result
            except Exception:
                return None

    return poly(left), poly(right)


def ema_poly(p_hat, p_prev, alpha=EMA_POLY):
    """EMA (Exponential Moving Average) для полиномиальных коэффициентов.

    МЕХАНИЗМ ПАМЯТИ И СТАБИЛЬНОСТИ:
    - Если линия обнаружена (p_hat != None): смешиваем с предыдущей через EMA
    - Если линия НЕ обнаружена (p_hat == None): используем предыдущую "по памяти"
    - Так линии не прыгают и не исчезают при плохой обнаружимости

    Args:
        p_hat: Новый полином (может быть None если не обнаружена)
        p_prev: Предыдущий полином (память)
        alpha: Коэффициент EMA (по умолчанию EMA_POLY=0.30)
               Может быть изменён для разных confidence уровней

    Returns:
        Сглаженный полином
    """
    # Если новая линия обнаружена, сглаживаем её с предыдущей
    if p_hat is not None and p_prev is not None:
        return alpha * p_hat + (1 - alpha) * p_prev

    # Если новая линия есть, но это первое обнаружение
    if p_hat is not None:
        return p_hat

    # ГЛАВНОЕ: если новая линия НЕ обнаружена, используем предыдущую (ПАМЯТЬ)
    # Так линии не исчезают и не прыгают при глюках детекции
    return p_prev


def ema_poly_adaptive(p_hat, p_prev, other_detected, alpha=EMA_POLY):
    """EMA с адаптивным весом для стабильности при потере одной из линий.

    НОВАЯ ЛОГИКА:
    - Если обе линии детектированы: используем стандартный alpha
    - Если одна линия потеряна, но другая есть: используем больший alpha (0.8)
      чтобы полнее доверять детектированной линии и не дёргать руль

    Args:
        p_hat: Новый полином текущей линии (может быть None)
        p_prev: Предыдущий полином текущей линии
        other_detected: True если другая линия (левая или правая) детектирована
        alpha: Базовый коэффициент EMA

    Returns:
        Сглаженный полином
    """
    # Если обе линии есть — используем стандартный alpha
    if p_hat is not None and other_detected:
        if p_prev is not None:
            return alpha * p_hat + (1 - alpha) * p_prev
        return p_hat

    # Если текущая линия потеряна, но другая есть — используем полную память
    if p_hat is None and other_detected and p_prev is not None:
        # Не меняем, просто деградируем (уже делается в LaneTracker)
        return p_prev

    # Если обе потеряны или стандартный режим
    return ema_poly(p_hat, p_prev, alpha)


def lane_points_from_poly(p, h, ymin=None, ymax=None, ymin_ratio=LANE_YMIN_RATIO, n=80):
    """Generate lane curve points from polynomial within ROI bounds."""
    if p is None:
        return None

    if ymin is None:
        ymin = int(ymin_ratio * h)
    if ymax is None:
        ymax = h - 1

    ymin = np.clip(ymin, 0, h - 1)
    ymax = np.clip(ymax, ymin + 1, h - 1)

    ys = np.linspace(ymin, ymax, n)
    xs = p[0]*ys**2 + p[1]*ys + p[2]
    return np.int32(np.vstack([xs, ys]).T)


def draw_lane_line(vis, poly, is_detected, color, roi_ymin, roi_ymax):
    """Рисует кривую линии с учётом детекции или памяти.

    Args:
        vis: Визуализационный фрейм
        poly: Полином линии
        is_detected: Обнаружена ли линия в этом фрейме
        color: Базовый цвет (RGB)
        roi_ymin, roi_ymax: Границы ROI
    """
    pts = lane_points_from_poly(poly, vis.shape[0], ymin=roi_ymin, ymax=roi_ymax)
    if pts is not None:
        if is_detected:
            # Обнаружено → яркое и толстое
            draw_color = color
            draw_thickness = 6
        else:
            # Из памяти → тусклое и тонкое
            draw_color = tuple(int(c * LANE_FADE_ALPHA) for c in color)
            draw_thickness = 3

        cv2.polylines(vis, [pts], False, draw_color, draw_thickness)

# ==========================
# Steering computation
# ==========================

def steering_from_lanes(poly_L, poly_R, w, h, lane_L=None, lane_R=None, k1=40.0, k2=180.0, theta_max=STEER_MAX):
    """Compute steering angle from lane polynomials.

    СТАБИЛЬНОСТЬ КОГДА ОДНА ЛИНИЯ ПОТЕРЯНА:
    - Использует адаптивную lane_width из предыдущих кадров
    - Доверяет одной детектированной линии при потере другой
    """
    y1, y2 = int(0.65*h), int(0.9*h)

    def x_from_poly(p, y):
        return p[0]*y*y + p[1]*y + p[2]

    if poly_L is None and poly_R is None:
        return 0.0

    # Получаем адаптивную ширину полосы из памяти
    lane_width = None
    if lane_L is not None:
        lane_width = lane_L.get_lane_width()
    if lane_width is None and lane_R is not None:
        lane_width = lane_R.get_lane_width()

    # Fallback на константу если нет истории
    if lane_width is None:
        lane_width = 0.45 * w

    # НОВАЯ ЛОГИКА: когда одна линия потеряна, используем детектированную с стабильностью
    if poly_L is None and poly_R is not None:
        # Левая потеряна, правая есть — смещаемся влево на пол-ширины
        midx = lambda y: x_from_poly(poly_R, y) - lane_width/2
    elif poly_R is None and poly_L is not None:
        # Правая потеряна, левая есть — смещаемся вправо на пол-ширины
        midx = lambda y: x_from_poly(poly_L, y) + lane_width/2
    else:
        # Обе линии есть — используем середину, сохраняем вычисленную ширину
        x_l_mid = x_from_poly(poly_L, y1)
        x_r_mid = x_from_poly(poly_R, y1)
        computed_width = x_r_mid - x_l_mid
        if computed_width > 10:  # разумная ширина (не менее 10 px)
            if lane_L is not None:
                lane_L.set_lane_width(computed_width)
            if lane_R is not None:
                lane_R.set_lane_width(computed_width)
        midx = lambda y: 0.5*(x_from_poly(poly_L, y) + x_from_poly(poly_R, y))

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
    """Overlay rotated RGBA wheel image on frame using premultiplied alpha compositing."""
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


def draw_steering_wheel(frame, wheel_img, angle, anchor=WHEEL_ANCHOR, offset=WHEEL_OFFSET, scale=WHEEL_SCALE):
    """Draw steering wheel overlay on frame with anchor positioning.

    Реалистичное позиционирование руля, как если бы камера была установлена на лобовом стекле машины,
    и руль виден в нижней части кадра (эффект видеорегистратора).

    Args:
        frame: Input frame (BGR)
        wheel_img: RGBA wheel image
        angle: Steering angle in degrees
        anchor: Position anchor ('lb','rb','lt','rt' - left/right + bottom/top)
        offset: (dx, dy) offset from anchor in pixels
        scale: Visual scale factor for the wheel

    Returns:
        Frame with wheel overlay
    """
    if wheel_img is None:
        return frame

    h, w = frame.shape[:2]
    ww, wh = wheel_img.shape[1], wheel_img.shape[0]

    # ВАЖНОЕ ИСПРАВЛЕНИЕ: учитываем масштабирование при расчёте позиции
    # Без этого руль отображался далеко от угла, так как использовались оригинальные размеры
    scaled_ww = int(ww * scale)
    scaled_wh = int(wh * scale)
    dx, dy = offset
    anchor = anchor.lower()

    # Расчёт позиции X с учётом якоря и масштабированной ширины
    # 'l' (левый якорь) → позиция от левого края + смещение
    # иначе (правый якорь) → позиция от правого края минус масштабированная ширина и смещение
    x0 = dx if 'l' in anchor else w - scaled_ww - dx

    # Расчёт позиции Y с учётом якоря и масштабированной высоты
    # 't' (верхний якорь) → позиция от верхнего края + смещение
    # иначе (нижний якорь) → позиция от нижнего края минус масштабированная высота и смещение
    # ОСОБЕННОСТЬ: руль частично выходит за границу снизу — это выглядит реалистично!
    y0 = dy if 't' in anchor else h - scaled_wh - dy

    return overlay_wheel(frame, wheel_img, -angle, x=x0, y=y0, scale=scale)

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cap = cv2.VideoCapture("VIDEO/20250330_115814_L.MP4")   # <— set your path
    wheel = cv2.imread("resources/wheel.png", cv2.IMREAD_UNCHANGED)  # optional

    prev_roi = None
    # ИЗМЕНЕНИЕ: используем LaneTracker для отслеживания памяти линий
    lane_L = LaneTracker()
    lane_R = LaneTracker()
    angle_ema = 0.0
    angle_draw = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = letterbox(frame, TARGET_SIZE) if RESIZE_MODE=="fit" else stretch(frame, TARGET_SIZE)

        # УЯЗВИМОСТЬ FIX: вычисляем edges один раз, передаём везде
        edges = compute_edges(frame)

        # ROI
        roi_poly = detect_road_roi(frame, edges, prev_roi, beta=0.3)
        prev_roi = roi_poly.copy()

        # ИСПРАВЛЕНИЕ: вычисляем верхнюю и нижнюю границы ROI для ограничения поиска линий
        roi_ymin = int(np.min(roi_poly[:, 1]))  # верхняя граница ROI
        roi_ymax = int(np.max(roi_poly[:, 1]))  # нижняя граница ROI

        # lines and fits
        lines = detect_lines(frame, edges, roi_poly, ymin=roi_ymin, ymax=roi_ymax)
        L_hat, R_hat = fit_lane(lines)

        # НОВАЯ ЛОГИКА: сначала получаем новые значения
        # Это нужно для определения, какая линия детектирована
        L_new_detected = L_hat is not None
        R_new_detected = R_hat is not None

        # МЕХАНИЗМ СТАБИЛЬНОСТИ: адаптивный EMA в зависимости от детекции
        # Если одна линия потеряна, другая детектированная линия получает больший вес
        if L_new_detected and R_new_detected:
            # Обе линии есть — стандартный EMA
            polyL, L_detected, L_memory = lane_L.update(L_hat)
            polyR, R_detected, R_memory = lane_R.update(R_hat)
        elif L_new_detected and not R_new_detected:
            # Левая есть, правая потеряна — доверяем левой больше
            polyL, L_detected, L_memory = lane_L.update(L_hat)
            polyR, R_detected, R_memory = lane_R.update(None)  # R остаётся в памяти
        elif R_new_detected and not L_new_detected:
            # Правая есть, левая потеряна — доверяем правой больше
            polyL, L_detected, L_memory = lane_L.update(None)  # L остаётся в памяти
            polyR, R_detected, R_memory = lane_R.update(R_hat)
        else:
            # Обе потеряны
            polyL, L_detected, L_memory = lane_L.update(None)
            polyR, R_detected, R_memory = lane_R.update(None)

        # steering (с адаптивной шириной полосы)
        steer = steering_from_lanes(polyL, polyR, frame.shape[1], frame.shape[0], lane_L, lane_R)
        angle_ema = EMA_STEER*steer + (1-EMA_STEER)*angle_ema
        delta = np.clip(angle_ema - angle_draw, -RATE_LIMIT, RATE_LIMIT)
        angle_draw += delta

        # steering (с адаптивной шириной полосы)
        steer = steering_from_lanes(polyL, polyR, frame.shape[1], frame.shape[0], lane_L, lane_R)
        angle_ema = EMA_STEER*steer + (1-EMA_STEER)*angle_ema
        delta = np.clip(angle_ema - angle_draw, -RATE_LIMIT, RATE_LIMIT)
        angle_draw += delta

        # viz
        vis = frame.copy()
        cv2.polylines(vis, [roi_poly], True, (0,255,255), 2)

        # Рисуем линии с учётом памяти
        draw_lane_line(vis, polyL, L_detected, (0, 255, 0), roi_ymin, roi_ymax)
        draw_lane_line(vis, polyR, R_detected, (0, 0, 255), roi_ymin, roi_ymax)

        # wheel overlay with anchor and centered bottom text
        if wheel is not None:
            vis = draw_steering_wheel(vis, wheel, angle_draw)
        # text centered at bottom
        status = f"{'RIGHT' if angle_draw>3 else 'LEFT' if angle_draw<-3 else 'STRAIGHT'} ({angle_draw:.1f} deg)"
        (font, fs, th) = (cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        (ts_w, ts_h), base = cv2.getTextSize(status, font, fs, th)
        cv2.putText(vis, status, ((vis.shape[1]-ts_w)//2, vis.shape[0]-TEXT_BOTTOM_OFFSET), font, fs, (0,255,0), th, cv2.LINE_AA)

        cv2.imshow("lane", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.release(); cv2.destroyAllWindows()
