# 2D_driving_sim
2D Driving Simulation implementing SAC and A star algorithm together on a custom gymnaisum enviornment. 

Coded SAC from scratch using a older youtube tutorial 
https://github.com/dia-agrawal/sac

https://www.youtube.com/watch?v=ioidsRlf79o

Updated gymnasium to work with current GPU (4070 Ti) because old github could only work on CPU. Older SAC used Value Network which isn't really implemented in modern SAC programs. 
Also incoperated SAC to have a learnable alpha parameter that doesn't allow alpha to get closer to 0 (So algorithm doesn't become determinstic and influence exploration to some extent at all times) 

## How the enviornment works 
<img width="787" height="517" alt="image" src="https://github.com/user-attachments/assets/ea1d491a-850b-4cc0-bb26-4ad1b56a3c58" />

### HOW THE ENVIORNMENT WORKS 
Randomly generated buildings (Yellow Blocks) with some randomized grid like roads added in 
Green is grass 
Dark grey are roads 

### WHAT CERTAIN OBJECTS REPRESENT 
Car looks like a triangle so we can see where its heading to 
The orange dot is the initial waypoint. Once the car reaches this waypoint, next one will appear
The X circuled is the final destination 

### REWARD 
Time penalty of -0.001 each step (so learns to use most optimal path to target) 
#### **TURN PENALTY ** (in the beginning car was just spinning in circles) _ 
turn_penalty = -0.01 * abs(yaw_delta)
reward += turn_penalty
#### STAND STILL PENALTY 
if self._agent_velocity < 0.5:
    reward += -0.02
#### DENSE GUIDANCE SIGNAL 
forward_reward = 1.0 * forward
reward += forward_reward
#### SPEED BONUS 
forward_reward = 1.0 * forward
reward += forward_reward
#### HEADING ALIGNMENT  
align_reward = k_align * (abs(head_err_prev) - abs(head_err_new))
reward += align_reward
#### ENVIORNMENT REWARD 
##### For buildings: 
reward += -10.0
terminated = True
##### For Grass: 
reward += -0.05
##### For Road
reward += 0.00
#### WAYPOINT REWARD 
reward += wp_reward      # 0.5
wp_idx += 1
#### FINAL WAYPOINT REWARD 
if not _final_wp_paid:
    reward += wp_reward      # 0.5
    reward += wp_final_bonus # 5.0
    _final_wp_paid = True
#### GOAL REWARD 
reward += 50.0
terminated = True

I have attached a video of my trained agent performing in the 2d simulation 
https://youtu.be/fKqXSSwY0y8

At the 1 minute mark, i show how the agent performs under no training (to verify I have used my own SAC algorithm and properly trained it) 

