import gym
import numpy as np
import random

def rbf(x, c, s):
    a = (x - c)
    a = a.reshape(2,1)
    at = a.reshape(1,2)
    b = at.dot(s)
    b = b.dot(a)
    return np.exp(- b * b)


length_position = 5
length_velocity = 3
centers_position = np.linspace(-1.2 , 0.6, length_position)
centers_velocity = np.linspace(-0.07 , 0.07 , length_velocity)
print(centers_position)
centers = []
for i in centers_position:
    for j in centers_velocity:
        centers.append((i,j))

actions = [0,1,2]
sigma = np.matrix('1 0;0 1')
class learn:
    def __init__(self, actions, centers):
        self.env = gym.make("MountainCar-v0")
        self.alpha = 0.004
        self.actions = actions
        self.centers = centers
        self.epsilon = 0.1
        self.landa = 1
        self.gama = 0.9
        self.weights = np.zeros((len(actions),len(centers) + 1))

    def generate_fi(self, state):
        fi = np.zeros(len(self.centers) + 1)
        for i in range(len(self.centers) + 1):
            if i == len(self.centers):
                fi[i] = 0.5*state[0] + 10 * state[1]
                continue
            fi[i] = rbf(state, np.array([self.centers[i][0], self.centers[i][1]]), sigma)

        fi = np.asarray(fi)
        return fi

    def generate_state_action_q(self, state, action):
        fi = self.generate_fi(state)
        theta = self.weights[self.actions.index(action)]
        return theta.T.dot(fi)

    def generate_state_qs(self, state):
        result = []
        for i in self.actions:
            result.append(self.generate_state_action_q(state, i))
        return result

    def max_index(self, x):
        m = max(x)
        indexes = [i for i, j in enumerate(x) if j == m]
        return random.choice(indexes)

    def choose_action(self, state):
        values = self.generate_state_qs(state)
        if random.random() > self.epsilon:
            chosen = self.max_index(values)
        else:
            chosen = random.randrange(len(values))


        return self.actions[chosen]

    def update(self):
        state = self.env.reset()
        chosen_action = self.choose_action(state)

        Zlast = np.zeros(len(self.centers) + 1).reshape(len(self.centers) + 1, 1)
        c = 0


        while state[0] < 0.5:
            c = c + 1
            if (c%100) == 0:
                print(c)
            observation, reward, done, info = self.env.step(chosen_action)
            self.env.render()
            if observation[0] > 0.49:
                reward = 100

            next_action = self.choose_action(observation)
            theta = self.weights[self.actions.index(chosen_action)]
            theta = theta.reshape(len(self.centers) + 1, 1)

            delta = reward + self.gama * self.generate_state_action_q(observation, next_action) - self.generate_state_action_q(state, chosen_action)
            fi = self.generate_fi(state).reshape(len(self.centers) + 1, 1)
            Znew = fi + self.gama * self.landa * Zlast
            theta = np.add(theta, self.alpha * (1  ) * delta * Znew)
            self.weights[self.actions.index(chosen_action)] = theta.reshape(len(self.centers) + 1)
            chosen_action = next_action
            state = observation

            Zlast = Znew

        print("it converged in {} steps".format(c))








def start(number):
    algo = learn(actions, centers)
    for i in range(number):
        algo.update()



start(3)
