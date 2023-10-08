# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def cross_product(p1, p2):
    return p1[0] * p2[1] - p1[1] * p2[0]

def cos(vec1, vec2):
    vec1_dis = np.linalg.norm(vec1)
    vec2_dis = np.linalg.norm(vec2)
    dot = dot_product(vec1, vec2)
    if vec1_dis != 0 and vec2 != 0:
        return  dot / (vec1_dis * vec2_dis)
    else:
        return dot

def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    center = alien.get_centroid()
    shape = alien.get_shape()
    if shape == "Ball":
        for i in walls:
            start_point = (i[0], i[1])
            end_point = (i[2], i[3])
            vector = [i[0] - i[2], i[1] - i[3]]
            dis = point_segment_distance(center, (start_point, end_point))
            max_dis = alien.get_width()
            if dis > max_dis:
                continue
            else:
                vec1 = [i[0] - center[0], i[1] - center[1]]
                vec2 = [i[2] - center[0], i[3] - center[1]]
                point1_center = np.linalg.norm(vec1)
                point2_center = np.linalg.norm(vec2)
                if point1_center <= max_dis or point2_center <= max_dis:
                    return True 
                else:
                    cos1 = cos(vec1, vector)
                    cos2 = - cos(vec2, vector)
                    if cos1 > 0 and cos2 > 0:
                        return True 
                    else:
                        continue
    elif shape == "Horizontal":
        head, tail = alien.get_head_and_tail()
        width = alien.get_width()
        edges = [((head[0] + width, head[1] - width), (head[0] + width, head[1] + width)), 
                ((head[0] + width, head[1] - width), (tail[0] - width, tail[1] - width)),
                ((tail[0] - width, tail[1] - width), (tail[0] - width, tail[1] + width)),
                ((head[0] + width, head[1] + width), (tail[0] - width, tail[1] + width))]
        polygon = [
            (head[0], head[1] - width),
            (head[0], head[1] + width),
            (tail[0], tail[1] + width),
            (tail[0], tail[1] - width)
        ]
        for wall in walls:
            if is_point_in_polygon((wall[0], wall[1]), polygon) or is_point_in_polygon((wall[2], wall[3]), polygon):
                    return True
            else:
                if point_segment_distance(head, ((wall[0], wall[1]), (wall[2], wall[3]))) <= width or point_segment_distance(tail, ((wall[0], wall[1]), (wall[2], wall[3]))) <= width:
                    return True
                else:
                    for edge in edges:
                        start_point = (wall[0], wall[1])
                        end_point = (wall[2], wall[3])
                        if do_segments_intersect((start_point, end_point), edge):
                            return True
                        else:
                            continue
    else:
        head, tail = alien.get_head_and_tail()
        width = alien.get_width()
        # head[1] += width 
        # tail[1] -= width
        edges = [((head[0] - width, head[1] - width), (head[0] + width, head[1] - width)), 
                ((head[0] - width, head[1] - width), (tail[0] - width, tail[1] + width)),
                ((tail[0] - width, tail[1] + width), (tail[0] + width, tail[1] + width)),
                ((head[0] + width, head[1] - width), (tail[0] + width, tail[1] + width))]
        polygon = [
            (head[0] + width, head[1]),
            (tail[0] + width, tail[1]),
            (tail[0] - width, tail[1]),
            (tail[0] - width, head[1])
        ]
        for wall in walls:
            if is_point_in_polygon((wall[0], wall[1]), polygon) or is_point_in_polygon((wall[2], wall[3]), polygon):
                    return True
            else:
                if point_segment_distance(head, ((wall[0], wall[1]), (wall[2], wall[3]))) <= width or point_segment_distance(tail, ((wall[0], wall[1]), (wall[2], wall[3]))) <= width:
                    return True
                else:
                    for edge in edges:
                        start_point = (wall[0], wall[1])
                        end_point = (wall[2], wall[3])
                        if do_segments_intersect((start_point, end_point), edge):
                            return True
                        else:
                            continue
    return False

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    center = alien.get_centroid()
    shape = alien.get_shape()
    # head, tail = alien.get_head_and_tail()
    window_ud_seg = [(0, 0, window[0], 0), (0, window[1], window[0], window[1])]
    window_lr_seg = [(0, 0, 0, window[1]), (window[0], 0, window[0], window[1])]
    for i in window_lr_seg:
        start_point = (i[0], i[1])
        end_point = (i[2], i[3])
        dis = point_segment_distance(center, (start_point, end_point))
        if shape == "Horizontal":
            max_dis = alien.get_length() / 2 + alien.get_width()
        else:
            max_dis = alien.get_width()
        if dis <= max_dis:
            return False 
        else:
            continue
    for i in window_ud_seg:
        start_point = (i[0], i[1])
        end_point = (i[2], i[3])
        dis = point_segment_distance(center, (start_point, end_point))
        if shape == "Vertical":
            max_dis = alien.get_length() / 2 + alien.get_width()
        else:
            max_dis = alien.get_width()
        if dis <= max_dis:
            return False
        else:
            continue 
    return True 

def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    symbol = 0
    # if the dot times of PA, PB, PC, PD with AB, BC, CD, DA is the same symbol,
    # then returns true
    for i in range(4):
        point1, point2 = polygon[i], polygon[(i + 1) % 4]
        vector = (point2[0] - point1[0], point2[1] - point1[1])
        vector2 = (point[0] - point1[0], point[1] - point1[1])
        dis = point_segment_distance(point, (point1, point2))
        if dis == 0:
            return True
        dot = vector[0] * vector2[0] + vector[1] * vector2[1]
        if dot >= 0:
            # symbol += 1
            continue
        else:
            return False
    return True

def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    if does_alien_touch_wall(alien, walls):
        return True
    cur_pos = alien.get_centroid()
    shape   = alien.get_shape()
    if cur_pos[0] == waypoint[0]:
        if cur_pos[1] == waypoint[1]:
            alien.set_alien_pos(waypoint)
            if does_alien_touch_wall(alien, walls):
                return True
            return False
        else:
            direction = 1
    elif cur_pos[1] == waypoint[1]:
        direction = 2
    else:
        direction = 0
    
    width = alien.get_width()
    length = alien.get_length() / 2 + alien.get_width()
    # move upwards and downwards
    if direction == 1:
        if shape == "Horizontal":
            polygon = [(cur_pos[0] - length, cur_pos[1]),
                       (cur_pos[0] + length, cur_pos[1]),
                       (waypoint[0] + length, waypoint[1]),
                       (waypoint[0] - length, waypoint[1])]
        else:
            polygon = [(cur_pos[0] - width, cur_pos[1]),
                       (cur_pos[0] + width, cur_pos[1]),
                       (waypoint[0] + width, waypoint[1]),
                       (waypoint[0] - width, waypoint[1])]
        edges = [
            (polygon[0], polygon[1]),
            (polygon[1], polygon[2]),
            (polygon[2], polygon[3]),
            (polygon[3], polygon[0]),
        ]
        for wall in walls:
                if is_point_in_polygon((wall[0], wall[1]), polygon) or is_point_in_polygon((wall[2], wall[3]), polygon):
                    return True
                else:
                    for edge in edges:
                        if do_segments_intersect(((wall[0], wall[1]), (wall[2], wall[3])), edge):
                            return True
                    continue
    # move leftwards and rightwards
    if direction == 2:
        if shape == "Vertical":
            polygon = [(cur_pos[0], cur_pos[1] - length),
                       (cur_pos[0], cur_pos[1] + length),
                       (waypoint[0], waypoint[1] + length),
                       (waypoint[0], waypoint[1] - length)]
            
        else:
            polygon = [(cur_pos[0], cur_pos[1] - width),
                       (cur_pos[0], cur_pos[1] + width),
                       (waypoint[0], waypoint[1] + width),
                       (waypoint[0], waypoint[1] - width)]
        edges = [
            (polygon[0], polygon[1]),
            (polygon[1], polygon[2]),
            (polygon[2], polygon[3]),
            (polygon[3], polygon[0]),
        ]
        for wall in walls:
                if is_point_in_polygon((wall[0], wall[1]), polygon) or is_point_in_polygon((wall[2], wall[3]), polygon):
                    return True
                else:
                    for edge in edges:
                        if do_segments_intersect(((wall[0], wall[1]), (wall[2], wall[3])), edge):
                            return True
                    continue
    if direction == 0:
        print("?")
        s1 = ((cur_pos[0], cur_pos[1]), (waypoint[0], waypoint[1]))
        for i in walls:
            start_point = (i[0], i[1])
            end_point = (i[2], i[3])
            if do_segments_intersect((start_point, end_point), s1) is False:
                continue
            else:
                return True
    alien.set_alien_pos(waypoint)
    if does_alien_touch_wall(alien, walls):
        alien.set_alien_pos(cur_pos)
        return True
    # print("?")
    # s1 = [cur_pos[0], cur_pos[1], waypoint[0], waypoint[1]]
    # for i in walls:
    #     if do_segments_intersect(i, s1) is False:
    #         continue
    #     else:
    #         return True
    alien.set_alien_pos(cur_pos)
    return False
    
def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    vector1 = [p[0] - s[1][0], p[1] - s[1][1]]
    vector = [s[1][0] - s[0][0], s[1][1] - s[0][1]]
            # calculate the distance of two points from the s2[0]
    vector2 = [p[0] - s[0][0], p[1] - s[0][1]]
    # dis = abs(vector2[0] * vector[1] - vector2[1] * vector[0])
    # length_vec2 = np.linalg.norm(vector2)
    # return dis / length_vec2
    dot_product1 = dot_product(vector, vector1)
    dot_product2 = dot_product(vector, vector2)
    # if their symbols are different => the projection is on the seg
    if dot_product1 * dot_product2 >= 0:
        dis1 = np.linalg.norm(vector1)
        dis2 = np.linalg.norm(vector2)
        dis = min(dis1, dis2)
        return dis
    else:
        dis = abs(vector[0] * vector2[1] - vector[1] * vector2[0])
        length_vec2 = np.linalg.norm(vector)
        return dis / length_vec2

def clock_rot(p1, p2):
    clk = cross_product(p1, p2)
    if clk == 0:
        return 0
    # clk wise
    elif clk > 0:
        return 1
    # unclk wise
    else:
        return -1

def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    # point 42
    s1_st = np.array([s1[0][0], s1[0][1]]); s1_end = np.array([s1[1][0], s1[1][1]])
    s2_st = np.array([s2[0][0], s2[0][1]]); s2_end = np.array([s2[1][0], s2[1][1]])
    # s1 + s2[0]
    p1 = s1_st - s2_st; p2 = s1_end - s2_st
    clk1 = clock_rot(p1, p2)
    on_line1 = point_segment_distance((s2[0][0], s2[0][1]), s1)
    # s1 + s2[1]
    p1 = s1_st - s2_end; p2 = s1_end - s2_end
    clk2 = clock_rot(p1, p2)
    on_line2 = point_segment_distance((s2[1][0], s2[1][1]), s1)
    
    # s2 + s1[1]
    p1 = s2_st - s1_end; p2 = s2_end - s1_end
    clk3 = clock_rot(p1, p2)
    on_line3 = point_segment_distance((s1[1][0], s1[1][1]), s2)
    
    # s2 + s1[0]
    p1 = s2_st - s1_st; p2 = s2_end - s1_st
    clk4 = clock_rot(p1, p2)
    on_line4 = point_segment_distance((s1[0][0], s1[0][1]), s2)
    
    if clk1 != clk2 and clk3 != clk4:
        return True 
    
    if (on_line1 == 0 and clk1 == 0) :
        return True
    if (on_line2 == 0 and clk2 == 0) :
        return True
    if (on_line3 == 0 and clk3 == 0) :
        return True
    if (on_line4 == 0 and clk4 == 0) :
        return True
    
    return False

def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:2
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0
    else:
        dis1 = 10000; dis2 = 10000
        for i in [s1[0], s1[1]]:
            # calculate the distance of two points from the s2[0]
            dis1 = min(dis1, point_segment_distance(i, s2))
        for i in [s2[0], s2[1]]:
            # calculate the distance of two points from the s2[0]
            dis2 = min(dis2, point_segment_distance(i, s1))
        return min(dis1, dis2)

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
