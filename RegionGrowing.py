import numpy as np

def is_close(pixel, target, diff):
    return abs(pixel - target) < diff
FORGET_IT = -1  # Define a constant for "forget" value

def pixel_label_and_check_neighbor(g, output, count, sum_, target, diff, stack, r, e, g_label):
    output[r, e] = g_label
    count[0] += 1
    sum_[0] += int(g[r, e])
    target[0] = sum_[0] / count[0]

    for dr in range(-1, 2):
        for de in range(-1, 2):
            nr, ne = r + dr, e + de
            if (0 <= nr < g.shape[0]) and (0 <= ne < g.shape[1]):
                if g[nr, ne] != FORGET_IT and output[nr, ne] == 0 and is_close(g[nr, ne], target[0], diff):
                    stack.append((nr, ne))
                    output[nr, ne] = g_label

def improved_region_growing(g, diff, min_area, max_area):
    g = g.astype(np.int32)
    rows, cols = g.shape
    output = np.zeros_like(g, dtype=int)
    g_label = 2

    for i in range(rows):
        for j in range(cols):
            if output[i, j] == 0 and g[i, j] != FORGET_IT:
                target = [float(g[i, j])]
                sum_ = [int(g[i, j])]
                count = [1]
                stack = [(i, j)]
                output[i, j] = g_label

                while stack:
                    r, e = stack.pop()
                    pixel_label_and_check_neighbor(g, output, count, sum_, target, diff, stack, r, e, g_label)

                if min_area <= count[0] <= max_area:
                    g_label += 1
                else:
                    output[output == g_label] = 0

    return output