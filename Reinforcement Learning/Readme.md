# Introduction
Implemmenting Q_Learning in snake game and training a model for eating snack and avioding from hitting other snake or walls.
# Usage
The npy files in Q_tables are pre_trained weights using different parameters(Three models). You can assign any Q-tables you want to SNAKE_1_Q_TABLE and SNAKE_2_Q_TABLE in constants.py
and fine-tune them or simply create two new Q-tables by not passing any file to the Snake class :

try:
    self.q_table = np.load(file_name)
except:
    self.q_table = np.zeros((2**16, 4)) # Initialize Q-table for a state space of 65536 states and 4 actions

