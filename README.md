# Mario-DeepQ-Learning
Play Mario using Deep Convolutional Q-Learning on Tensorflow

I really like to thanks a lot to [justinmeister](https://github.com/justinmeister/Mario-Level-1) for this awesome pygame code for Mario.

I just create Deep Q-learning interface using Tensorflow on this Mario Pygame

![screenshot](https://raw.github.com/justinmeister/Mario-Level-1/master/screenshot.png)

to run, just simply
```bash
python mario_level_1.py
```
But before that, checkout hyper-parameters on data/model.py first
```python
# {A, left, right}
ACTIONS = 3

# probability action to press
PRESS_THRESHOLD = 0.5

# constant if learn the wrong thing
GAMMA = 0.99

# step to learn
OBSERVE = 10000
EXPLORE = 20000

# constants to do random decision
FINAL_EPSILON = 0.001
# initial should between 0.5 - 0.9, it will decay over time to do random actions
INITIAL_EPSILON = 0.99

# memory space to hold
REPLAY_MEMORY_SIZE = 50000

# batch size
BATCH = 32
FRAME_PER_ACTION = 1
```

![screenshot](https://raw.githubusercontent.com/huseinzol05/Mario-DeepQ-Learning/master/screenshotmario/1.png)
![screenshot](https://raw.githubusercontent.com/huseinzol05/Mario-DeepQ-Learning/master/screenshotmario/2.png)
![screenshot](https://raw.githubusercontent.com/huseinzol05/Mario-DeepQ-Learning/master/screenshotmario/3.png)
