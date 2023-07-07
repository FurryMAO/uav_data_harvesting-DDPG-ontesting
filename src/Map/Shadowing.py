import numpy as np
import os
import tqdm
from src.Map.Map import load_map


def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    if obstacles[y0, x0]:
        return
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[y0, x0] = False

    while x0 != x1 or y0 != y1:
        if 2 * error - y_dist > x_dist - 2 * error:
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[y0, x0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[y0, x0] = False


def calculate_shadowing(map_path, save_as):
    total_map = load_map(map_path)
    obstacles = total_map.obstacles
    size = total_map.obstacles.shape[0]
    total = size * size

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    with tqdm.tqdm(total=total) as pbar:
        for i, j in np.ndindex(total_map.obstacles.shape):
            shadow_map = np.ones((size, size), dtype=bool)

            for x in range(size):
                bresenham(i, j, x, 0, obstacles, shadow_map)
                bresenham(i, j, x, size - 1, obstacles, shadow_map)
                bresenham(i, j, 0, x, obstacles, shadow_map)
                bresenham(i, j, size - 1, x, obstacles, shadow_map)

            total_shadow_map[j, i] = shadow_map
            pbar.update(1)

    np.save(save_as, total_shadow_map)
    return total_shadow_map

########################################
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class BoundingBox:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


def is_intersection(point_a, point_b, obstacle):
    # 构造射线
    ray_start = point_a
    ray_direction = Point(point_b.x - point_a.x, point_b.y - point_a.y)

    # 检查射线与边界盒相交
    if ray_intersects_box(ray_start, ray_direction, obstacle):
        # 进一步检查射线与障碍物的具体形状是否相交
        if ray_intersects_obstacle(ray_start, ray_direction, obstacle):
            return True

    return False


def ray_intersects_box(ray_start, ray_direction, box):
    tmin = (box.x_min - ray_start.x) / ray_direction.x
    tmax = (box.x_max - ray_start.x) / ray_direction.x

    if tmin > tmax:
        tmin, tmax = tmax, tmin

    tymin = (box.y_min - ray_start.y) / ray_direction.y
    tymax = (box.y_max - ray_start.y) / ray_direction.y

    if tymin > tymax:
        tymin, tymax = tymax, tymin

    if tmin > tymax or tymin > tmax:
        return False

    if tymin > tmin:
        tmin = tymin

    if tymax < tmax:
        tmax = tymax

    return tmin > 0


def ray_intersects_obstacle(ray_start, ray_direction, obstacle):
    # 实现障碍物形状与射线相交的具体算法，这取决于障碍物的形状表示方法
    # 可以使用线段与多边形相交的算法等

    # 这里只是一个示例，假设障碍物是一个矩形
    obstacle_box = BoundingBox(obstacle.x_min, obstacle.y_min, obstacle.x_max, obstacle.y_max)
    return ray_intersects_box(ray_start, ray_direction, obstacle_box)


# 示例使用
obstacle1 = BoundingBox(1, 1, 3, 3)
obstacle2 = BoundingBox(4, 4, 6, 6)

point1 = Point(0, 0)
point2 = Point(2, 2)
point3 = Point(5, 5)

print(is_intersection(point1, point2, obstacle1))  # True，射线与障碍物1相交
print(is_intersection(point2, point3, obstacle1))  # False，射线与障碍物1不相交
print(is_intersection(point2, point3, obstacle2))  # False，射线与障碍物2不相交


def load_or_create_shadowing(map_path):
    shadow_file_name = os.path.splitext(map_path)[0] + "_shadowing.npy"
    if os.path.exists(shadow_file_name):
        return np.load(shadow_file_name)
    else:
        return calculate_shadowing(map_path, shadow_file_name)
