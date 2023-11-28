import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = state 
        self.N[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action] += 1
        return       

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        prime_food_dir_x, prime_food_dir_y, prime_adjoin_wall_x, prime_adjoin_wall_y, prime_adjoin_body_top, prime_adjoin_body_bottom, prime_adjoin_body_left, prime_adjoin_body_right = s
        food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = s_prime
        N = self.N[prime_food_dir_x, prime_food_dir_y, prime_adjoin_wall_x, prime_adjoin_wall_y, prime_adjoin_body_top, prime_adjoin_body_bottom, prime_adjoin_body_left, prime_adjoin_body_right, a] 
        learning_rate = self.C / (self.C + N) 
        prime_Q = self.Q[prime_food_dir_x, prime_food_dir_y, prime_adjoin_wall_x, prime_adjoin_wall_y, prime_adjoin_body_top, prime_adjoin_body_bottom, prime_adjoin_body_left, prime_adjoin_body_right, a]
        Q_max = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, utils.RIGHT]
        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP,  ]:
            Q = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
            Q_max = max(Q, Q_max)
        Q_update = prime_Q + learning_rate * (r + self.gamma * (Q_max) - prime_Q)
        self.Q[prime_food_dir_x, prime_food_dir_y, prime_adjoin_wall_x, prime_adjoin_wall_y, prime_adjoin_body_top, prime_adjoin_body_bottom, prime_adjoin_body_left, prime_adjoin_body_right, a] = Q_update
        return
    def collide_obstacle(self, previous_state):
        adjoin_wall_x, adjoin_wall_y = previous_state[2], previous_state[3]
        if adjoin_wall_x == 1 and self.a == 2:
            return True
        elif adjoin_wall_x == 2 and self.a == 3:
            return True
        elif adjoin_wall_y == 1 and self.a == 0:
            return True
        elif adjoin_wall_y == 2 and self.a == 1:
            return True
        else:
            return False 
    def collide_body(self, previous_state):
        adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = previous_state[4], previous_state[5], previous_state[6], previous_state[7]
        # up = 0
        if self.a == 0 and adjoin_body_top:
            return True 
        # Down = 1
        elif self.a == 1 and adjoin_body_bottom:
            return True
        # Left = 2
        elif self.a == 2 and adjoin_body_left:
            return True
        # Right = 3
        elif self.a == 3 and adjoin_body_right:
            return True
        else:
            return False
    # def collision(self, points, env, state):
    #     if self.collide_body(points, state):
    #         return True
    def choose_action(self, s_prime):
        opt_action = utils.RIGHT
        food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right = s_prime
        opt_f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, opt_action]
        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP,  ]:
            if self._train:
                visit_times = self.N[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
                if visit_times < self.Ne:
                    f = 1 
                else:
                    f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
            else:
                f = self.Q[food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right, action]
            if f > opt_f:
                opt_action = action
                opt_f = f
        return opt_action
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        # TODO - MP12: write your function here
        # * calculate the reward of the action
        # snake_head_x, snake_head_y = environment[0], environment[1]
        # snake_body = environment[2]
        # food_x, food_y = environment[3], environment[4]
        # rock_x, rock_y = environment[5], environment[6]
        # * Generalize the current state
        s_current = self.generate_state(environment)
        # * If training
        if self._train:  
            if self.s is not None and self.a is not None:
            # * Generate reward
                if points == self.points + 1:
                    reward = 1
                elif dead:         
                    reward = -1
                else:
                    reward = -0.1
                self.update_n(self.s, self.a)
                self.update_q(self.s, self.a, reward, s_current)
            
            # * update internal state variables
            if dead:
                self.reset()
                action = np.random.choice(self.actions)
            else:
                self.s = s_current
                # * choose the optimize action
                action = self.choose_action(s_current)
                self.a = action  
                self.points = points
        else:
            self.a = self.choose_action(s_current)
        return self.a

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        snake_head_x, snake_head_y = environment[0], environment[1]
        snake_body = environment[2]
        food_x, food_y = environment[3], environment[4]
        rock_x, rock_y = environment[5], environment[6]
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        # food_dir_x, food_dir_y
        if food_x < snake_head_x:
            food_dir_x = 1
        elif food_x > snake_head_x:
            food_dir_x = 2
        else:
            food_dir_x = 0
        if food_y < snake_head_y:
            food_dir_y = 1
        elif food_y > snake_head_y:
            food_dir_y = 2
        else:
            food_dir_y = 0
        # adjoining_all_x, adjoining_all_y
        # movable range: x: [1, self.display_weight - 2], movable range y: [1, self.display_height - 2]
        # rock: [rockx, rocky], [rockx + 1, rocky]
        if snake_head_x == 1 or (snake_head_x == rock_x + 2 and snake_head_y == rock_y): 
            adjoin_wall_x = 1
        elif snake_head_x == self.display_width - 2 or (snake_head_x == rock_x - 1 and snake_head_y == rock_y):
            adjoin_wall_x = 2
        else:
            adjoin_wall_x = 0
        if snake_head_y == 1 or ((snake_head_x == rock_x  or snake_head_x == rock_x + 1) and snake_head_y == rock_y + 1): 
            adjoin_wall_y = 1
        elif snake_head_y == self.display_height - 2 or ((snake_head_x == rock_x  or snake_head_x == rock_x + 1) and snake_head_y == rock_y - 1):
            adjoin_wall_y = 2
        else:
            adjoin_wall_y = 0
        # adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoin_body_left = 1
        else:
            adjoin_body_left = 0
        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoin_body_right = 1
        else:
            adjoin_body_right = 0
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoin_body_top = 1
        else:
            adjoin_body_top = 0
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoin_body_bottom = 1
        else:
            adjoin_body_bottom = 0
        state = (food_dir_x, food_dir_y, adjoin_wall_x, adjoin_wall_y, adjoin_body_top, adjoin_body_bottom, adjoin_body_left, adjoin_body_right)
        return state
