import gym
import numpy as np
import random


def rbf(x, c, s):
    xtemp = np.array([ (x[0] + 1.2) /1.8, (x[1] + 0.07) / 0.14])



    a = (xtemp - c)
    a = a.reshape(2,1)
    at = a.reshape(1,2)
    b = at.dot(s)
    b = b.dot(a)

    return np.exp(- b * b)

def rbf_action(x, c, s, action):
    xtemp = np.array([(x[0] + 1.2) / 1.8, (x[1] + 0.07) / 0.14, action])

    a = (xtemp - c)
    a = a.reshape(3, 1)
    at = a.reshape(1, 3)
    b = at.dot(s)
    b = b.dot(a)

    return np.exp(- b * b)




length_position = 5
length_velocity = 3
centers_position = np.linspace(0 , 1, length_position)
centers_velocity = np.linspace(0 , 1 , length_velocity)
centers_of_action = [0.5, 1.5]
centers = []
centers_action = []
for i in centers_position:
    for j in centers_velocity:
        centers.append((i,j))
        for k in centers_of_action:
            centers_action.append((i, j, k))

actions = [0,1,2]
sigma = np.matrix('1 0;0 1')
sigma2 = np.matrix('1 0 0;0 1 0;0 0 2')
class learn:
    def __init__(self, actions, centers, centers_action):
        self.env = gym.make("MountainCar-v0")
        self.alpha_theta = 0.0035
        self.alpha_w = 0.0035
        self.actions = actions
        self.centers = centers
        self.centers_action = centers_action
        self.gama = 1
        self.theta = np.zeros(len(self.centers_action)).reshape(len(self.centers_action) , 1)
        self.weights_w = np.zeros(len(self.centers)).reshape(len(self.centers), 1)

    def generate_fi(self, state, action):
        fi = np.zeros(len(self.centers_action))
        for i in range(len(self.centers_action)):
            fi[i] = rbf_action(state, np.array([self.centers_action[i][0], self.centers_action[i][1], self.centers_action[i][2]]), sigma2, action)


        fi = np.asarray(fi)
        return fi
    def generate_fi_s(self,state):
        fi = np.zeros(len(self.centers))
        for i in range(len(self.centers)):
            fi[i] = rbf(state, np.array([self.centers[i][0], self.centers[i][1]]), sigma)
        fi = np.asarray(fi)
        return fi
    def generate_state_action_h(self, state, action):
        fi = self.generate_fi(state,action)
        return self.theta.T.dot(fi)

    def generate_state_hs(self, state):
        result = []
        for i in self.actions:
            result.append(self.generate_state_action_h(state, i))
        return result

    def max_index(self, x):
        m = max(x)
        indexes = [i for i, j in enumerate(x) if j == m]
        return random.choice(indexes)

    def get_prefrences(self, state):
        values = self.generate_state_hs(state)
        preference = [np.exp(x) for x in values]
        return preference


    def choose_action(self, state):
        values = self.generate_state_hs(state)
        preference = self.get_prefrences(state)
        sumation = sum(preference)
        prob = [float(x) / float(sumation) for x in preference]
        chosen = np.random.choice(len(values), 1, prob)
        return self.actions[chosen[0]]

    def generate_value(self, state):
        fi = self.generate_fi_s(state).reshape(len(self.centers), 1)
        return self.weights_w.T.dot(fi)

    def policy_value(self, state, action):
        h_s_a = self.generate_state_action_h(state,action)
        preference = self.get_prefrences(state)
        sumation = sum(preference)
        return np.exp(h_s_a) / sumation

    def gradient_of_policy(self, state, action):
        fi_s_a = self.generate_fi(state, action)
        sum = 0
        for i in self.actions:
            sum = sum + self.policy_value(state, action) * self.generate_fi(state, i)
        return (fi_s_a - sum).reshape(len(self.centers_action), 1)






    def update(self):
        state = self.env.reset()
        chosen_action = self.choose_action(state)

        #self.env.render()

        c = 0


        gamma_generator = 1

        while state[0] < 0.5:
            c = c + 1
            if (c%1000) == 0:
                print(c)
            observation, reward, done, info = self.env.step(chosen_action)
            #self.env.render()
            if observation[0] > 0.49:
                reward = 100

            next_action = self.choose_action(observation)
            state_value = self.generate_value(state)
            next_value = self.generate_value(observation)
            delta = reward + self.gama * next_value - state_value
            fi = self.generate_fi_s(state).reshape(len(self.centers), 1)
            self.weights_w = self.weights_w + self.alpha_w * delta * gamma_generator * fi
            gradient = self.gradient_of_policy(state, chosen_action)
            self.theta = self.theta + self.alpha_theta * gamma_generator * delta * gradient

            gamma_generator = gamma_generator * self.gama



            chosen_action = next_action
            state = observation




        print("it converged in {} steps".format(c))








def start(number):
    algo = learn(actions, centers, centers_action)
    for i in range(number):
        algo.update()



start(3)
