from geometry import is_point_in_polygon

center = [
    [10, 5],
    [10, 10],
    [2, 0],
    [0, 2]
]

polygon = [
    [(0, 0), (10, 0), (10, 10), (0, 10)],
    [(0, 0), (10, 0), (10, 10), (0, 10)],
    [(0, 0), (1, 0), (3, 0), (4, 0)],
    [(0, 0), (0, 1), (0, 3), (0, 4)]
]

for i, j in zip(center, polygon):
    res = is_point_in_polygon(i, j)
