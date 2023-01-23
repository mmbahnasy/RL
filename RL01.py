import numpy as np
import matplotlib.pyplot as plt


ACTIONS = {'R': (1, 0), 'L': (-1, 0), 'D': (0, -1), 'U': (0, 1)}

WorldDim = 5

class Enviroment():
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.state = (0, 0)
        self.steps = 0

    def step(self, action):
        x, y = self.state
        x += ACTIONS[action][0]
        y += ACTIONS[action][1]
        self.state = (x, y)
        self.steps += 1
        reward = -1
        isDone = False
        if self.state == (WorldDim-1, WorldDim-1):
            reward = 0
            isDone = True
        return self.state, reward, isDone

    def getAllowedActions(self, state):
        ''' Return list of keys of the ACTION dictionary '''
        allowedActions = []
        for action in ACTIONS:
            if self.isValidAction(state, action):
                allowedActions.append(action)
        return allowedActions

    def isValidAction(self, state, action):
        x, y = state
        x += ACTIONS[action][0]
        y += ACTIONS[action][1]
        if x<0 or y<0 or x>=WorldDim or y>=WorldDim:
            return False
        return True


class Agent():
    def __init__(self, env: Enviroment, gamma=0.99, explorationFactor=0.8) -> None:
        self.env = env
        self.gamma = gamma
        self.explorationFactor = explorationFactor
        self.QTable = np.zeros((WorldDim, WorldDim))
        self.stateHistory = [] # list of (state, reward) tuple

    def selectAction(self, state, allowedActions):
        nextAction = None
        randomNum = np.random.random()
        if randomNum < self.explorationFactor:
            # explor all possibple actions
            nextAction = np.random.choice(allowedActions)
        else:
            # Exploitation: select the action with the meximum possible Q value for the next state:
            possibleQValue  = [self.QTable[state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1]] for action in allowedActions]
            nextAction = allowedActions[np.array(possibleQValue).argmax(0)]
        return nextAction

    def updateQTable(self):
        accumulatedReward = 0
        for currState, reward in reversed(self.stateHistory):
            # self.QTable[currState[0], currState[1]] = reward + self.gamma * self.QTable[currState[0], currState[1]]
            self.QTable[currState[0], currState[1]] += self.gamma * (accumulatedReward - self.QTable[currState[0], currState[1]])
            accumulatedReward += reward
        self.stateHistory = []

if __name__ == '__main__':
    env = Enviroment()
    robot = Agent(env, gamma=0.9, explorationFactor=0.25)
    moveHistory = []

    for i in range(3000):
        isDone = False
        while not isDone:
            action = robot.selectAction(env.state, env.getAllowedActions(env.state))
            env.state, reward, isDone = env.step(action) # update the env according to the action
            robot.stateHistory.append((env.state, reward)) # update the robot memory with state and reward
            if env.steps > 100:
                # Terminate the training if it took too long to find the goal
                break

        robot.updateQTable()
        robot.explorationFactor -= 10e-5 # decrease the exploration factor after each episode
        moveHistory.append(env.steps) # get a history of number of steps taken to plot later
        env.reset() # reinitialize the env

    print(robot.QTable.round(decimals=2))
    plt.plot(moveHistory, "b--")
    plt.grid(True)
    plt.show()
