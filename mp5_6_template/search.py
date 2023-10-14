# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq
from state import MazeState

# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    # return {
    #     "astar": astar,
    # }.get(searchMethod, [])(maze)
    states_explored, path = astar(maze)
    maze.states_explored = states_explored
    return path 


# TODO: VI
def astar(maze):
    start_point = maze.get_start()
    visited_states = {start_point: (None, 0)}
    frontier = []
    heapq.heappush(frontier, start_point)
    state_tmp = 0
    while len(frontier) > 0:
        visited_pos = heapq.heappop(frontier)
        state_tmp += 1
        if visited_pos.is_goal() == True:
            path = backtrack(visited_states, visited_pos)
            visited_pos.maze.states_explored = state_tmp - 1
            # maze = visited_pos.maze
            return visited_pos.maze.states_explored, path
        
        visited_pos.maze.states_explored = state_tmp
        visited_pos.dist_from_start = visited_states[visited_pos][1]
        print(visited_pos, visited_pos.maze.states_explored)
        neighbors = visited_pos.get_neighbors()
        # state_tmp = visited_pos.maze.states_explored
        
        for i in neighbors: 
            if i not in visited_states:
                heapq.heappush(frontier, i)
                visited_states[i] = (visited_pos, i.dist_from_start)
            else:
                if i.dist_from_start < visited_states[i][1]:
                    visited_states[i] = (visited_pos, i.dist_from_start)
    return visited_pos.maze.states_explored, None


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []
    path.append(current_state)
    # Your code here ---------------
    parent_node, distant = visited_states[current_state]
    while distant != 0:
        path.append(parent_node)
        parent_node, distant = visited_states[parent_node]
    path = path[::-1]
    # ------------------------------
    return path