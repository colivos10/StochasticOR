# %%
import numpy as np

# %%
door_time = {0: 3, 1: 5, 2: 7}
door_index = [0, 1, 2]
time = 0
time_saver = []
np.random.seed(0)
# %% Smart Strategy
time = 0
for i in range(100000):
    while True:
        door_selected = np.random.choice(door_index, 1)
        if door_selected == 0:
            time = time + 3
            break
        elif door_selected == 1:
            time = time + 5
            door_index.remove(1)
        else:
            time = time + 7
            door_index.remove(2)
    time_saver.append(time)
    time = 0
    door_index = [0, 1, 2]

print(sum(time_saver)/len(time_saver))

# %% Slightly Dump
time = 0
for i in range(100000):
    while True:
        door_selected = np.random.choice(door_index, 1)
        if i > 0:
            door_index.append(door_to_remove)
        door_to_remove = door_selected[0]
        door_index.remove(door_to_remove)
        if door_selected[0] == 0:
            time = time + 3
            break
        elif door_selected[0] == 1:
            time = time + 5
        else:
            time = time + 7
    time_saver.append(time)
    time = 0
    door_index = [0, 1, 2]

print(sum(time_saver)/len(time_saver))

# %% Dump Strategy
time = 0
for i in range(100000):
    while True:
        door_selected = np.random.choice(door_index, 1)
        if door_selected == 0:
            time = time + 3
            break
        elif door_selected == 1:
            time = time + 5
        else:
            time = time + 7
    time_saver.append(time)
    time = 0
    door_index = [0, 1, 2]

print(sum(time_saver)/len(time_saver))