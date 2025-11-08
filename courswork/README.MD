# Road Detection Project

This
project
implements
computer
vision
algorithms
for road detection and region of interest (ROI) extraction in driving videos and images.

## Features

- Image and video loading/processing
- Adaptive ROI detection for road areas
    - Hood/dashboard boundary estimation
- Vanishing
point
detection
for perspective analysis
    - Masking
of
non - road
regions

## Requirements

- Python 3.11+
- Poetry (package manager)
- OpenCV(cv2)
- NumPy
- Matplotlib

## Installation

1. Clone the repository
2. Install Poetry if not already installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Activate the virtual environment:
   ```bash
   poetry shell
   ```
