import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}
    
    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------
    while len(frontier) > 0:
        visit_node = heapq.heappop(frontier)
        # * if the state is the min: found
        if visit_node.is_goal() == True:
            path = backtrack(visited_states, visit_node)
            return path
        # * if not, then add son of the state in the frontier(state and step)
        #visited_states[visit_node.state] = (visit_node.state, visit_node.dist_from_start)
        neighbors = visit_node.get_neighbors()
        for i in neighbors:
            if i not in visited_states:
                heapq.heappush(frontier, i)
                # print("New vist puzzle: {} step: {}, h: {}".format(i, i.dist_from_start, i.h))
                visited_states[i] = (visit_node, i.dist_from_start)
            else:
            # ! if it is encountered second times, its value should be renewed, if the path is shorter
                if i.dist_from_start < visited_states[i][1]:
                    visited_states[i] = (visit_node, i.dist_from_start)
            # else:
            #     if i.dist_from_start > visited_states[i][1]:
            #         visited_states[i] = (visit_node, i.dist_from_start)
    # ------------------------------
    # if you do not find the goal return an empty list
    return []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    path.append(goal_state)
    # Your code here ---------------
    parent_node, distant = visited_states[goal_state]
    while distant != 0:
        path.append(parent_node)
        parent_node, distant = visited_states[parent_node]
    # ------------------------------
    return path