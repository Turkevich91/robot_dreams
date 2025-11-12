# Configuration Reference Guide

## 🎛️ Полный справочник всех параметров

---

## 📐 Основные размеры и режимы

```python
TARGET_SIZE = (1280, 720)   # Целевой размер кадра (ширина, высота)
RESIZE_MODE = "fit"         # "fit" (letterbox) или "stretch"
```

| Параметр | Значение | Назначение | Рекомендация |
|----------|----------|-----------|--------------|
| TARGET_SIZE | (1280, 720) | Стандартный размер обработки | Больше = точнее, но медленнее |
| RESIZE_MODE | "fit" | Сохранять aspect ratio | Используйте "fit" для HD видео |

---

## 🔍 Обнаружение краёв (Canny Edge Detection)

```python
CANNY_LOW = 60              # Нижний порог градиента
CANNY_HIGH = 180            # Верхний порог градиента
```

| Параметр | Значение | Диапазон | Эффект |
|----------|----------|----------|--------|
| CANNY_LOW | 60 | 30-80 | ↑ меньше = больше слабых краёв |
| CANNY_HIGH | 180 | 100-250 | ↑ больше = только сильные края |

**Примеры:**
- Плохая видимость (туман): CANNY_LOW=80, CANNY_HIGH=200
- Хорошая видимость: CANNY_LOW=50, CANNY_HIGH=150
- Шумное видео: CANNY_LOW=100, CANNY_HIGH=220

---

## 📏 Линии (Hough Transform)

```python
HOUGH_THRESH = 30          # Порог голосов для детекции
HOUGH_MINLEN = 40          # Минимальная длина отрезка (px)
HOUGH_MAXGAP = 50          # Максимальный разрыв между точками (px)
```

| Параметр | Значение | Диапазон | Эффект |
|----------|----------|----------|--------|
| HOUGH_THRESH | 30 | 10-100 | ↑ больше = менее чувствительно |
| HOUGH_MINLEN | 40 | 20-100 | ↑ больше = только длинные линии |
| HOUGH_MAXGAP | 50 | 20-100 | ↑ больше = соединяет разрывы |

**Примеры:**
- Пунктирные линии (дорога): HOUGH_MINLEN=30, HOUGH_MAXGAP=70
- Сплошные линии: HOUGH_MINLEN=60, HOUGH_MAXGAP=30
- Шумное видео: HOUGH_THRESH=50, HOUGH_MINLEN=80

---

## 🎚️ Сглаживание (EMA Coefficients)

```python
EMA_POLY = 0.30             # Вес полинома: 30% новое + 70% память
EMA_STEER = 0.05            # Вес угла: 5% новое + 95% память
RATE_LIMIT = 1.5            # Макс изменение угла за фрейм (град)
STEER_MAX = 25.0            # Максимальный угол руля (град)
```

| Параметр | Значение | Диапазон | Эффект |
|----------|----------|----------|--------|
| EMA_POLY | 0.30 | 0.05-0.80 | ↑ больше = менее плавное, быстрее реагирует |
| EMA_STEER | 0.05 | 0.01-0.20 | ↑ больше = менее плавное руль движется быстрее |
| RATE_LIMIT | 1.5 | 0.5-5.0 | ↑ больше = быстрее меняет угол |
| STEER_MAX | 25.0 | 15-45 | ↑ больше = более резкие повороты |

**Примеры:**
- Очень плавное вождение: EMA_STEER=0.02, RATE_LIMIT=0.5
- Нормальное вождение: EMA_STEER=0.05, RATE_LIMIT=1.5
- Активное вождение: EMA_STEER=0.15, RATE_LIMIT=3.0

---

## 💾 Память линий (Lane Memory)

```python
LANE_MEMORY_FRAMES = 20     # Макс фреймов хранения линии
LANE_FADE_ALPHA = 0.3       # Прозрачность линии из памяти (0.0-1.0)
LANE_POLY_DECAY = 0.98      # Деградация полинома (-2% в фрейм)
```

| Параметр | Значение | Диапазон | Эффект |
|----------|----------|----------|--------|
| LANE_MEMORY_FRAMES | 20 | 5-60 | ↑ больше = дольше помнит (но может залипнуть) |
| LANE_FADE_ALPHA | 0.3 | 0.1-0.8 | ↑ больше = ярче отображается линия памяти |
| LANE_POLY_DECAY | 0.98 | 0.90-0.99 | ↑ меньше = быстрее забывает |

**Примеры:**
- Стабильная дорога: LANE_MEMORY_FRAMES=40, LANE_POLY_DECAY=0.97
- Нестабильная дорога: LANE_MEMORY_FRAMES=10, LANE_POLY_DECAY=0.95
- Смена полос: LANE_MEMORY_FRAMES=5, LANE_POLY_DECAY=0.90

**Расчёт времени памяти:**
```
Время памяти = LANE_MEMORY_FRAMES / FPS
При 30 FPS и LANE_MEMORY_FRAMES=20 → 0.67 секунд
```

---

## 🗺️ ROI (Region of Interest)

```python
SKY_CROP = 0.35             # Исключить верхние 35% (небо)
HOOD_CROP = 0.315           # Исключить нижние 31.5% (капот)
ROI_TOP_HALF_WIDTH_RATIO = 0.30  # Ширина верхней части ROI (доля ширины)
BOTTOM_MARGIN_X_RATIO = 0.03     # Отступ от края внизу
LANE_YMIN_RATIO = 0.55      # Начало отрисовки кривых линий
```

| Параметр | Значение | Использование |
|----------|----------|---------------|
| SKY_CROP | 0.35 | Исключает небо (35% сверху) |
| HOOD_CROP | 0.315 | Исключает капот (31.5% снизу) |
| ROI_TOP_HALF_WIDTH_RATIO | 0.30 | Сужение ROI к vanishing point |
| BOTTOM_MARGIN_X_RATIO | 0.03 | Отступ от левого/правого края |
| LANE_YMIN_RATIO | 0.55 | Где начинать рисовать линии |

**Примеры ROI для разных машин:**
```python
# Низко расположенная камера
SKY_CROP = 0.40
HOOD_CROP = 0.25

# Высоко расположенная камера (лобовое стекло)
SKY_CROP = 0.25
HOOD_CROP = 0.40

# Узкий угол зрения
ROI_TOP_HALF_WIDTH_RATIO = 0.20

# Широкий угол зрения
ROI_TOP_HALF_WIDTH_RATIO = 0.40
```

---

## 🎛️ Отрисовка руля (Steering Wheel)

```python
WHEEL_ANCHOR = "lb"         # Якорь: l/r (лево/право) + b/t (внизу/вверху)
WHEEL_OFFSET = (20, 20)     # Смещение от якоря (dx, dy) в px
WHEEL_SCALE = 0.60          # Масштаб руля (0.0-1.0)
TEXT_BOTTOM_OFFSET = 28     # Расстояние текста от дна
```

**Якори позиционирования:**
```
'lb' = нижний левый (left-bottom)
'rb' = нижний правый (right-bottom)
'lt' = верхний левый (left-top)
'rt' = верхний правый (right-top)
```

**Примеры позиционирования:**
```python
# Нижний левый угол (стандарт)
WHEEL_ANCHOR = "lb"
WHEEL_OFFSET = (20, 20)
WHEEL_SCALE = 0.60

# Нижний правый угол
WHEEL_ANCHOR = "rb"
WHEEL_OFFSET = (20, 20)
WHEEL_SCALE = 0.60

# Меньший размер
WHEEL_SCALE = 0.40

# Больший размер
WHEEL_SCALE = 0.80
```

---

## 🔧 Внутренние параметры алгоритма

```python
# RANSAC для vanishing point (не рекомендуется менять)
# find_vanishing_point_ransac(iterations=100, threshold=40.0, min_inliers=3)

# Фильтрация углов наклона (в fit_lane)
# 0.5 < |k| < 3.0  ← строгая фильтрация

# Морфологический kernel (в detect_lines)
# kernel_rect = (15, 3)  ← для скоростной трассы

# Outlier detection (в fit_lane)
# 2.5 * sigma  ← порог отброса выбросов
```

---

## 📊 Рекомендуемые комбинации параметров

### 🌤️ Хорошая погода, хайвей, прямая дорога
```python
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_THRESH = 20
EMA_STEER = 0.03        # Очень плавно
RATE_LIMIT = 1.0
LANE_MEMORY_FRAMES = 30 # Долгая память
```

### 🌧️ Дождь, слабая видимость
```python
CANNY_LOW, CANNY_HIGH = 80, 220
HOUGH_THRESH = 50
EMA_STEER = 0.10        # Быстрее реагирует
RATE_LIMIT = 2.0
LANE_MEMORY_FRAMES = 10 # Короче память
```

### 🌆 Город, извилистые дороги
```python
CANNY_LOW, CANNY_HIGH = 60, 180
HOUGH_THRESH = 30
HOUGH_MINLEN = 30       # Короче линии
HOUGH_MAXGAP = 70       # Больше разрывов
EMA_STEER = 0.08
RATE_LIMIT = 2.0
```

### 🏎️ Спортивная езда, быстрые повороты
```python
CANNY_LOW, CANNY_HIGH = 60, 180
HOUGH_THRESH = 25
EMA_STEER = 0.15        # Быстрее
RATE_LIMIT = 3.0
STEER_MAX = 35.0        # Более резко
LANE_MEMORY_FRAMES = 5  # Быстро забывает
```

### 🚛 Грузовик, широкие полосы
```python
ROI_TOP_HALF_WIDTH_RATIO = 0.40  # Шире ROI
LANE_MEMORY_FRAMES = 40          # Долгая память
EMA_POLY = 0.20                  # Медленнее реагирует
```

---

## 🎯 Быстрая оптимизация

### Проблема: Медленная обработка
```python
# ❌ Было
TARGET_SIZE = (1280, 720)

# ✅ Стало
TARGET_SIZE = (960, 540)
HOUGH_THRESH = 40  # Меньше чувствительность
```

### Проблема: Руль прыгает
```python
# ❌ Было
EMA_STEER = 0.15

# ✅ Стало
EMA_STEER = 0.03
RATE_LIMIT = 0.5
```

### Проблема: Потеря полос на пунктирных линиях
```python
# ❌ Было
HOUGH_MINLEN = 40
HOUGH_MAXGAP = 50

# ✅ Стало
HOUGH_MINLEN = 25
HOUGH_MAXGAP = 80
```

### Проблема: В ROI попадает небо
```python
# ❌ Было
SKY_CROP = 0.35

# ✅ Стало
SKY_CROP = 0.45
```

---

## 📝 Создание кастомного профиля

Сохраните в `config_custom.py`:

```python
# config_custom.py - My Custom Profile
TARGET_SIZE = (960, 540)
CANNY_LOW, CANNY_HIGH = 70, 200
HOUGH_THRESH = 35
EMA_STEER = 0.07
LANE_MEMORY_FRAMES = 15
SKY_CROP = 0.40
WHEEL_ANCHOR = "rb"
```

И используйте в `main.py`:
```python
from config_custom import *
```

---

**Последнее обновление:** Ноябрь 2025

