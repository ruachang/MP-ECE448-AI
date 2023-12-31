import copy
import math
from itertools import count

# NOTE: using this global index means that if we solve multiple
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()


# TODO VI
# Euclidean distance between two state tuples, of the form (x,y, shape)
def euclidean_distance(a, b):
    dis = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
    return math.sqrt(dis)

from abc import ABC, abstractmethod

class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0., use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    # To search a space we will iteratively call self.get_neighbors()
    # Return a list of State objects
    @abstractmethod
    def get_neighbors(self):
        pass

    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass

    # A* requires we compute a heuristic from eahc state
    # compute_heuristic should depend on self.state and self.goal
    # Return a float
    @abstractmethod
    def compute_heuristic(self):
        pass

    # The "less than" method ensures that states are comparable
    #   meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.dist_from_start + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    # __hash__ method allow us to keep track of which
    #   states have been visited before in a dictionary
    # You should hash states based on self.state (and sometimes self.goal, if it can change)
    # Return a float
    @abstractmethod
    def __hash__(self):
        pass

    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass


# State: a length 3 list indicating the current location in the grid and the shape
# Goal: a tuple of locations in the grid that have not yet been reached
#   NOTE: it is more efficient to store this as a binary string...
# maze: a maze object (deals with checking collision with walls...)
class MazeState(AbstractState):
    def __init__(self, state, goal, dist_from_start, maze, use_heuristic=True):
        # NOTE: it is technically more efficient to store both the mst_cache and the maze_neighbors functions globally,
        #       or in the search function, but this is ultimately not very inefficient memory-wise
        self.maze = maze
        self.maze_neighbors = maze.get_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)

    # TODO VI
    def get_neighbors(self):
        # if the shape changes, it will have a const cost of 10.
        # otherwise, the move cost will be the euclidean distance between the start and the end positions
        config = self.maze.alien.get_config()
        k_valid_neighbors = self.maze.get_neighbors(config[0], config[1], self.maze.alien.get_shape_idx())
        ori_shape = self.maze.alien.get_shape_idx()
        #ori_alien = self.maze.alien
        nbr_states = []
        
        for i in k_valid_neighbors:
            if ori_shape != i[2]:
                dis = self.dist_from_start + 10
            else:
                dis = self.dist_from_start + euclidean_distance(self.state, i)
            # have the same shape
            ori_maze = copy.deepcopy(self.maze)
            new_alien = ori_maze.create_new_alien(i[0], i[1], i[2])
            ori_maze.alien = new_alien
            new_ele = MazeState(
                state = (i[0], i[1], i[2]),
                goal = self.goal,
                dist_from_start = dis,
                maze = ori_maze,
                use_heuristic = self.use_heuristic
            )
            # new_ele.maze.alien = new_alien
            nbr_states.append(new_ele)
        # self.maze.alien = ori_alien
        return nbr_states

    # TODO VI
    def is_goal(self):
        if (self.state[0], self.state[1]) in self.goal:
            return True

    # We hash BOTH the state and the remaining goals
    #   This is because (x, y, h, (goal A, goal B)) is different from (x, y, h, (goal A))
    #   In the latter we've already visited goal B, changing the nature of the remaining search
    # NOTE: the order of the goals in self.goal matters, needs to remain consistent
    # TODO VI
    def __hash__(self):
        return hash(self.state) + hash(self.goal)

    # TODO VI
    def __eq__(self, other):
        return self.state == other.state and self.goal == other.goal

    # Our heuristic is: distance(self.state, nearest_goal)
    # We euclidean distance
    # TODO VI
    def compute_heuristic(self):
        min_dis = 10000
        for i in self.goal:
            min_dis = min(min_dis, euclidean_distance(i, (self.state[0], self.state[1])))
        return min_dis

    # This method allows the heap to sort States according to f = g + h value
    # TODO VI
    def __lt__(self, other):
        if self.dist_from_start + self.h < other.dist_from_start + other.h:
                return True
        else:
            if self.dist_from_start + self.h == other.dist_from_start + other.h:
                if self.tiebreak_idx < other.tiebreak_idx:
                    return True 
                else:
                    return False
            else:
                return False

    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)
