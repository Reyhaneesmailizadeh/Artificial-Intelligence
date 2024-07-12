from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np


class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((2**16, 4)) # Initialize Q-table for a state space of 65536 states and 4 actions

        self.lr = 0.1  
        self.discount_factor = 1 
        self.epsilon = 0.1 
        self.epsilon_decay = 0.9 
        self.min_epsilon = 0.05

    def get_optimal_policy(self, state):
        return np.argmax(self.q_table[state])


    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def create_state(self, snack, other_snake):
        # Wall dangers
        danger_wall_left = 1 if self.head.pos[0] == 0 else 0
        danger_wall_right = 1 if self.head.pos[0] == ROWS - 1 else 0
        danger_wall_up = 1 if self.head.pos[1] == 0 else 0
        danger_wall_down = 1 if self.head.pos[1] == ROWS - 1 else 0

        # Other snake dangers
        danger_snake_left = 1 if any(c.pos == (self.head.pos[0] - 1, self.head.pos[1]) for c in other_snake.body) else 0
        danger_snake_right = 1 if any(c.pos == (self.head.pos[0] + 1, self.head.pos[1]) for c in other_snake.body) else 0
        danger_snake_up = 1 if any(c.pos == (self.head.pos[0], self.head.pos[1] - 1) for c in other_snake.body) else 0
        danger_snake_down = 1 if any(c.pos == (self.head.pos[0], self.head.pos[1] + 1) for c in other_snake.body) else 0

        # Snack positions
        snack_pos = snack.pos
        snack_north = 1 if snack_pos[1] < self.head.pos[1] else 0
        snack_south = 1 if snack_pos[1] > self.head.pos[1] else 0
        snack_west = 1 if snack_pos[0] < self.head.pos[0] else 0
        snack_east = 1 if snack_pos[0] > self.head.pos[0] else 0

        # Current direction
        direction_up = 1 if self.dirny == -1 else 0
        direction_down = 1 if self.dirny == 1 else 0
        direction_left = 1 if self.dirnx == -1 else 0
        direction_right = 1 if self.dirnx == 1 else 0

        state = (danger_wall_left, danger_wall_right, danger_wall_up, danger_wall_down,
                danger_snake_left, danger_snake_right, danger_snake_up, danger_snake_down,
                snack_north, snack_south, snack_west, snack_east,
                direction_up, direction_down, direction_left, direction_right)
        
        state_index = sum([bit * (2 ** i) for i, bit in enumerate(state)])
        return state_index


    def move(self, snack, other_snake):
        state = self.create_state(snack, other_snake)
        action = self.make_action(state)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        next_state = self.create_state(snack, other_snake)
        
        return state, next_state, action
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        draw = False
        win_self, win_other = False, False
        
        if self.check_out_of_board():
            reward = reward -1000 # Punish the snake for getting out of the board
            win_other = True
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward = reward + 200# Reward the snake for eating
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward = reward -1000 #Punish the snake for hitting itself
            win_other = True
            reset(self, other_snake)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                reward = reward -1000 #  Punish the snake for hitting the other snake
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    reward = reward + 200 # Reward the snake for hitting the head of the other snake and being longer
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    reward = 0 # No winner
                    draw = True
                    # print(f"draw {draw}")
                else:
                    reward = reward - 1000 #Punish the snake for hitting the head of the other snake and being shorter
                    win_other = True
                    
            reset(self, other_snake)
            
        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay



    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        
