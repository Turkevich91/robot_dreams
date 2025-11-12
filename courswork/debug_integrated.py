"""
INTEGRATED DEBUG MODE
Режим отладки встроенный в main.py с возможностью:
- Переключения фильтров (1-9)
- Переключения элементов (r=ROI, w=wheel, t=text)
- Навигации по видео (стрелки, Shift, Ctrl)
- Запуска/паузы видео (SPACE)
"""

import cv2
import numpy as np
import math
from pathlib import Path
from datetime import datetime

# Импортируем основной код
from main import (
    letterbox, compute_edges, detect_road_roi, detect_lines, fit_lane, resize_frame,
    TARGET_SIZE, RESIZE_MODE, CANNY_LOW, CANNY_HIGH,
    HOUGH_THRESH, HOUGH_MINLEN, HOUGH_MAXGAP,
    EMA_POLY, EMA_STEER, RATE_LIMIT, STEER_MAX,
    LANE_YMIN_RATIO, LANE_MEMORY_FRAMES, LANE_POLY_DECAY,
    SKY_CROP, HOOD_CROP, lane_points_from_poly,
    draw_steering_wheel, overlay_wheel,
    WHEEL_SCALE, WHEEL_ANCHOR, WHEEL_OFFSET,
    TEXT_BOTTOM_OFFSET
)

# ========================
# DEBUG CONFIG
# ========================

# ФИЛЬТРЫ - базовое изображение (только одно может быть активно)
DEBUG_FILTERS = {
    '1': ('ORIGINAL', 'Исходное RGB видео'),
    '2': ('GRAY', 'Серый кадр (после Blur)'),
    '3': ('CANNY', 'Canny edges'),
}

# ВСПОМОГАТЕЛЬНЫЕ ЛИНИИ - оверлеи поверх фильтра (можно включать/выключать)
DEBUG_OVERLAYS = {
    'r': ('ROI', 'ROI полигон'),
    'h': ('HOUGH', 'Hough линии (жёлтые)'),
    'c': ('CLASSIFICATION', 'LEFT(зелёная) RIGHT(красная)'),
    'p': ('POLYFIT', 'Подогнанные кривые'),
    'w': ('WHEEL', 'Руль'),
    't': ('TEXT', 'Информационный текст'),
}

class DebugMode:
    """Интегрированный режим отладки"""

    def __init__(self, video_path, wheel_image_path=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise Exception(f"❌ Не удалось открыть видео: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_num = 0
        self.paused = True

        # ФИЛЬТР - только один активен
        self.current_filter = '3'  # По умолчанию CANNY

        # ВСПОМОГАТЕЛЬНЫЕ ЛИНИИ - можно комбинировать
        self.show_overlays = {
            'r': True,   # ROI
            'h': False,  # HOUGH
            'c': False,  # CLASSIFICATION
            'p': False,  # POLYFIT
            'w': False,  # WHEEL
            't': True,   # TEXT
        }

        # Загружаем изображение руля
        self.wheel_img = None
        if wheel_image_path and Path(wheel_image_path).exists():
            self.wheel_img = cv2.imread(wheel_image_path, cv2.IMREAD_UNCHANGED)

        # Lane tracking
        from main import LaneTracker
        self.lane_L = LaneTracker()
        self.lane_R = LaneTracker()
        self.angle_ema = 0.0
        self.angle_draw = 0.0  # Текущий угол руля для отрисовки

        # Инициализируем папку скриншотов
        self.screenshots_dir = self._init_screenshots_dir()

        print("\n" + "="*70)
        print("🔍 INTEGRATED DEBUG MODE - v2 (FILTERS + OVERLAYS)")
        print("="*70)
        self.print_help()

    def _init_screenshots_dir(self):
        """Создать папку для скриншотов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshots_dir = Path("debug_screenshots") / timestamp
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        print(f"📷 Скриншоты будут сохранены в: {screenshots_dir}\n")
        return screenshots_dir

    def save_screenshot(self, frame, display):
        """Сохранить скриншот текущего кадра"""
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # HH:MM:SS_milliseconds
        frame_num_padded = f"{self.frame_num:06d}"

        filter_name = DEBUG_FILTERS.get(self.current_filter, ('?', '?'))[0]
        overlays = [k for k in self.show_overlays if self.show_overlays[k] and k in DEBUG_OVERLAYS]
        overlays_str = "_".join(overlays) if overlays else "none"

        filename = self.screenshots_dir / f"{frame_num_padded}_{filter_name}_{overlays_str}_{timestamp}.png"

        cv2.imwrite(str(filename), display)
        print(f"📸 Скриншот сохранён: {filename.name}")

    def print_help(self):
        """Показать справку"""
        print("\n📺 ФИЛЬТРЫ (только один может быть активен):")
        for key, (name, desc) in DEBUG_FILTERS.items():
            marker = "→" if key == self.current_filter else " "
            print(f"  {marker} {key}: {name:15} - {desc}")

        print("\n👁️  ВСПОМОГАТЕЛЬНЫЕ ЛИНИИ (поверх фильтра, можно комбинировать):")
        for key, (name, desc) in DEBUG_OVERLAYS.items():
            status = "ON " if self.show_overlays[key] else "OFF"
            print(f"  {key} - {name:15} [{status}] - {desc}")

        print("\n⌨️  НАВИГАЦИЯ:")
        print("  [ / ]           - 1 кадр назад/вперёд")
        print("  ; / '           - 1 секунда назад/вперёд")
        print("  , / .           - 10 секунд назад/вперёд")
        print("  0               - начало видео")
        print("  SPACE           - пауза/воспроизведение")

        print("\n📷 СКРИНШОТЫ:")
        print("  s - сохранить текущий кадр")

        print("\n📋 ДРУГОЕ:")
        print("  h - показать справку")
        print("  ESC - выход")
        print("="*70 + "\n")

    def set_frame(self, frame_num):
        """Установить номер кадра"""
        self.frame_num = max(0, min(self.total_frames - 1, frame_num))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)

    def render_frame(self, frame):
        """Отрендерить кадр: ФИЛЬТР + вспомогательные линии"""

        # ============ ОБРАБОТКА ============
        frame_resized = resize_frame(frame, TARGET_SIZE, RESIZE_MODE)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, L2gradient=True)
        roi_poly = detect_road_roi(frame_resized, edges, beta=0.3)

        h, w = frame_resized.shape[:2]
        roi_ymin = int(np.min(roi_poly[:, 1]))
        roi_ymax = int(np.max(roi_poly[:, 1]))

        # Детектируем линии один раз для всех операций
        lines = detect_lines(frame_resized, edges, roi_poly, ymin=roi_ymin, ymax=roi_ymax)
        result = fit_lane(lines)
        if len(result) == 4:
            L_hat, R_hat, conf_L, conf_R = result
        else:
            L_hat, R_hat = result
            conf_L = conf_R = 0.0

        # ============ ВЫЧИСЛЯЕМ УГОЛ ПОВОРОТА (независимо от флагов) ============
        # Этот угол используется для отрисовки руля
        if L_hat is not None or R_hat is not None:
            y_ref = roi_ymax  # Нижняя точка ROI (где мы находимся)

            if L_hat is not None and R_hat is not None:
                # Если обе линии есть - берём среднее
                # dy/dx = 2*a*y + b (производная полинома)
                slope_L = 2 * L_hat[0] * y_ref + L_hat[1]
                slope_R = 2 * R_hat[0] * y_ref + R_hat[1]
                avg_slope = (slope_L + slope_R) / 2
            elif L_hat is not None:
                # Если только левая - используем её
                avg_slope = 2 * L_hat[0] * y_ref + L_hat[1]
            else:
                # Если только правая - используем её
                avg_slope = 2 * R_hat[0] * y_ref + R_hat[1]

            # Преобразуем уклон в угол (в градусах)
            # Угол = arctan(уклон) * 180/pi
            angle_rad = np.arctan(avg_slope)
            angle_deg = np.degrees(angle_rad)

            # Применяем EMA сглаживание для плавности
            self.angle_ema = EMA_STEER * angle_deg + (1 - EMA_STEER) * self.angle_ema

            # Ограничиваем угол в разумные пределы
            self.angle_draw = np.clip(self.angle_ema, -STEER_MAX, STEER_MAX)

        # ============ ВЫБИРАЕМ БАЗОВЫЙ ФИЛЬТР ============
        if self.current_filter == '1':  # ORIGINAL
            display = frame_resized.copy()
        elif self.current_filter == '2':  # GRAY
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif self.current_filter == '3':  # CANNY
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            display = frame_resized.copy()

        # ============ НАКЛАДЫВАЕМ ВСПОМОГАТЕЛЬНЫЕ ЛИНИИ ============

        # ROI
        if self.show_overlays['r']:
            cv2.polylines(display, [roi_poly], True, (0, 255, 255), 2)

        # HOUGH линии
        if self.show_overlays['h']:
            lines = detect_lines(frame_resized, edges, roi_poly)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # CLASSIFICATION (LEFT/RIGHT)
        if self.show_overlays['c']:
            lines = detect_lines(frame_resized, edges, roi_poly)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 == x1:
                        continue
                    k = (y2 - y1) / (x2 - x1)
                    if k < 0:
                        cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелёная - LEFT
                    else:
                        cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Красная - RIGHT

        # POLYFIT (подогнанные кривые)
        if self.show_overlays['p']:
            if L_hat is not None:
                ys = np.linspace(roi_ymin, roi_ymax, 100)
                xs = L_hat[0]*ys**2 + L_hat[1]*ys + L_hat[2]
                pts = np.int32(np.vstack([xs, ys]).T)
                cv2.polylines(display, [pts], False, (0, 255, 0), 3)

            if R_hat is not None:
                ys = np.linspace(roi_ymin, roi_ymax, 100)
                xs = R_hat[0]*ys**2 + R_hat[1]*ys + R_hat[2]
                pts = np.int32(np.vstack([xs, ys]).T)
                cv2.polylines(display, [pts], False, (0, 0, 255), 3)

        # WHEEL (руль)
        if self.show_overlays['w'] and self.wheel_img is not None:
            display = draw_steering_wheel(display, self.wheel_img, self.angle_draw,
                                         anchor=WHEEL_ANCHOR, offset=WHEEL_OFFSET, scale=WHEEL_SCALE)

        # TEXT (информационный текст)
        status = "⏸ PAUSE" if self.paused else "▶ PLAY"
        timestamp = self.frame_num / self.fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        millisecs = int((timestamp % 1) * 1000)

        filter_name = DEBUG_FILTERS.get(self.current_filter, ('?', '?'))[0]
        overlays_active = [DEBUG_OVERLAYS[k][0] for k in self.show_overlays if self.show_overlays[k] and k in DEBUG_OVERLAYS]
        overlays_str = " + ".join(overlays_active) if overlays_active else "none"

        info = f"{status} {filter_name:15} [{overlays_str}] | F:{self.frame_num:5d}/{self.total_frames} | {minutes:02d}:{seconds:02d}.{millisecs:03d}"

        if self.show_overlays['t']:
            cv2.putText(display, info, (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def handle_key(self, key):
        """Обработать нажатие клавиши"""

        if key == 255 or key == -1:
            # Ни одна клавиша не нажата
            return True

        if key == 27:  # ESC - выход
            return False

        elif key == ord('h') or key == ord('H'):
            self.print_help()

        elif key == ord(' '):
            self.paused = not self.paused
            status = "⏸ PAUSE" if self.paused else "▶ PLAY"
            print(f"  {status}")

        # НАВИГАЦИЯ - КАДРЫ [ / ]
        elif key == ord('['):  # [ - 1 кадр назад
            self.set_frame(self.frame_num - 1)
            print(f"  ← Frame {self.frame_num}")

        elif key == ord(']'):  # ] - 1 кадр вперёд
            self.set_frame(self.frame_num + 1)
            print(f"  → Frame {self.frame_num}")

        # НАВИГАЦИЯ - СЕКУНДЫ ; / '
        elif key == ord(';'):  # ; - 1 секунда назад
            self.set_frame(self.frame_num - int(self.fps))
            print(f"  ←← -1 second (frame {self.frame_num})")

        elif key == ord("'"):  # ' - 1 секунда вперёд
            self.set_frame(self.frame_num + int(self.fps))
            print(f"  →→ +1 second (frame {self.frame_num})")

        # НАВИГАЦИЯ - 10 СЕКУНД , / .
        elif key == ord(','):  # , - 10 секунд назад
            self.set_frame(self.frame_num - int(self.fps * 10))
            print(f"  ←←← -10 seconds (frame {self.frame_num})")

        elif key == ord('.'):  # . - 10 секунд вперёд
            self.set_frame(self.frame_num + int(self.fps * 10))
            print(f"  →→→ +10 seconds (frame {self.frame_num})")

        # НАВИГАЦИЯ - НАЧАЛО 0
        elif key == ord('0'):  # 0 - начало видео
            self.set_frame(0)
            print(f"  ⏮ Start (frame 0)")

        # ============ ФИЛЬТРЫ (1-3) ============
        elif key in [ord(str(i)) for i in range(1, 4)]:
            filter_key = str(chr(key))
            if filter_key in DEBUG_FILTERS:
                self.current_filter = filter_key
                print(f"  Filter: {DEBUG_FILTERS[filter_key][0]}")

        # ============ ВСПОМОГАТЕЛЬНЫЕ ЛИНИИ ============
        elif chr(key) in self.show_overlays:
            overlay_key = chr(key)
            self.show_overlays[overlay_key] = not self.show_overlays[overlay_key]
            status = "ON" if self.show_overlays[overlay_key] else "OFF"
            print(f"  {DEBUG_OVERLAYS[overlay_key][0]}: {status}")

        # ============ СКРИНШОТЫ ============
        elif key == ord('s') or key == ord('S'):
            return "screenshot"

        return True  # Продолжать

    def run(self):
        """Главный цикл"""

        print("▶ Запуск видео в режиме отладки\n")

        while True:
            # Загружаем/отображаем кадр
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("\n✅ Конец видео")
                    self.paused = True
                    self.set_frame(self.total_frames - 1)
                    continue
                self.frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
                ret, frame = self.cap.read()
                if not ret:
                    break

            # Отрендерить
            display = self.render_frame(frame)

            # Показать
            cv2.imshow("Debug Mode - Press 'h' for help", display)

            # Ждать клавиши
            wait_time = 1 if not self.paused else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key != 255:  # Если нажата клавиша
                result = self.handle_key(key)
                if result == "screenshot":
                    # Сохранить скриншот
                    self.save_screenshot(frame, display)
                elif result is False:
                    # Выход
                    break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Режим отладки завершён\n")


if __name__ == "__main__":
    try:
        debug = DebugMode(
            "VIDEO/20250331_184236_L.MP4",
            # "VIDEO/20250330_115814_L.MP4",
            "resources/wheel.png"
        )
        debug.run()
    except Exception as e:
        print(f"❌ Ошибка: {e}")

