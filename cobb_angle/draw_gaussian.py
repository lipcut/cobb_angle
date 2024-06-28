import numpy as np


def calculate_gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    def solve_quadratic(a, b, c):
        discriminant = np.sqrt(b**2 - 4 * a * c)
        return (-b + discriminant) / (2 * a)

    r1 = solve_quadratic(
        1, -(height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
    )
    r2 = solve_quadratic(4, -2 * (height + width), (1 - min_overlap) * width * height)
    r3 = solve_quadratic(
        4 * min_overlap,
        2 * min_overlap * (height + width),
        (min_overlap - 1) * width * height,
    )

    return min(r1, r2, r3)


def create_gaussian_2d(shape, sigma=1):
    y, x = [np.arange(-(s - 1) / 2, (s + 1) / 2) for s in shape]
    xx, yy = np.meshgrid(x, y)

    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def apply_gaussian_to_heatmap(heatmap, center, radius, k=1):
    diameter = int(2 * radius + 1)
    gaussian = create_gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = map(int, center)
    height, width = heatmap.shape[:2]

    left, right = max(0, x - radius), min(width, x + radius + 1)
    top, bottom = max(0, y - radius), min(height, y + radius + 1)

    gaussian_left = max(0, radius - x)
    gaussian_top = max(0, radius - y)

    gaussian_crop = gaussian[
        gaussian_top : gaussian_top + (bottom - top),
        gaussian_left : gaussian_left + (right - left),
    ]

    heatmap_crop = heatmap[top:bottom, left:right]
    np.maximum(heatmap_crop, gaussian_crop * k, out=heatmap_crop)

    return heatmap
