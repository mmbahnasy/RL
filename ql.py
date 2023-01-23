import numpy as np
import numpy as np
import matplotlib.pyplot as plt

ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class Maze(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.maze = np.zeros((6, 6))
        self.state = (0, 0)
        self.steps = 0

    def getNextState(state, action):
        state[0] += ACTIONS[action][0]
        state[1] += ACTIONS[action][1]
        if state[0] < 0: state[0] = 0
        if state[0] > 5: state[0] = 5
        if state[1] < 0: state[1] = 0
        if state[1] > 5: state[1] = 5
        return state

    def step(self, action):
        self.state = Maze.getNextState(self.state, action)
        self.steps += 1 # add steps
        reward = 0 if self.state == (5, 5) else -1
        isDone = True if self.state == (5, 5) else False
        return self.state, reward, isDone


class Agent(object):
    def __init__(self, states, alpha=0.15, randomFactor=0.2): # 80% explore, 20% exploit
        self.stateHistory = [((0, 0), 0)] # state, reward
        self.alpha = alpha
        self.randomFactor = randomFactor
        self.Qtable = {}
        self.initReward(states)

    def initReward(self, states):
        for i, row in enumerate(states):
            for j, col in enumerate(row):
                self.Qtable[(j, i)] = np.random.uniform(low=1.0, high=0.1)
    
    def choose_action(self, state):
        next_move = None
        randomN = np.random.random()
        if randomN < self.randomFactor:
            # if random number below random factor, choose random action
            next_move = np.random.choice(list(ACTIONS.keys()))
        else:
            # if exploiting, gather all possible actions and choose one with the highest Qtable (reward)
            possibleRewards = [self.Qtable[Maze.getNextState(state, action)] for action in ACTIONS]
            next_move = ACTIONS[np.array(possibleRewards).argmax(0)]

        return next_move

    def updateQValue(self):
        target = 0
        for prev, reward in reversed(self.stateHistory):
            self.Qtable[prev] = self.Qtable[prev] + self.alpha * (target - self.Qtable[prev])
            target += reward
        self.stateHistory = []


if __name__ == '__main__':
    maze = Maze()
    robot = Agent(maze.maze, alpha=0.1, randomFactor=0.25)
    moveHistory = []

    for i in range(5000):
        isDone = False
        while not isDone:
            action = robot.choose_action(maze.state) # choose an action (explore or exploit)
            state, reward, isDone = maze.step(action) # update the maze according to the action
            robot.stateHistory.append((state, reward)) # update the robot memory with state and reward
            if maze.steps > 1000:
                # end the robot if it takes too long to find the goal
                maze.state = (5, 5)
        
        robot.updateQValue() # robot should learn after every episode
        robot.randomFactor -= 10e-5 # decrease the exploration factor each episode of play
        moveHistory.append(maze.steps) # get a history of number of steps taken to plot later
        maze.reset() # reinitialize the maze

    print(robot.Qtable)
    plt.semilogy(moveHistory, "b--")
    plt.grid(True)
    plt.show()
