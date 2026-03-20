# 2D_driving_sim
2D Driving Simulation implementing SAC and A star algorithm together on a custom gymnaisum enviornment. 

Coded SAC from scratch using a older youtube tutorial 
https://github.com/dia-agrawal/sac

https://www.youtube.com/watch?v=ioidsRlf79o

Updated gymnasium to work with current GPU (4070 Ti) because old github could only work on CPU. Older SAC used Value Network which isn't really implemented in modern SAC programs. 
Also incoperated SAC to have a learnable alpha parameter that doesn't allow alpha to get closer to 0 (So algorithm doesn't become determinstic and influence exploration to some extent at all times) 

I have attached a video of my trained agent performing in the 2d simulation 
https://youtu.be/fKqXSSwY0y8

At the 1 minute mark, i show how the agent performs under no training (to verify I have used my own SAC algorithm and properly trained it) 

