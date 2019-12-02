import random
import numpy as np

p = 0.5
q = 0.6
l = 0.6


def arm_1_reward():
    prob = random.random()
    if prob < p :
        return np.random.normal(60, 8)

    return np.random.normal(-40, 8)


def arm_2_reward():
    prob = random.random()

    if prob < q:
        return np.random.uniform(40, 60)

    return np.random.uniform(-70, -40)


def arm_3_reward():
    prob = random.random()

    if prob < l:
        return np.random.normal(20, 8)

    return np.random.uniform(-10,10)


class epsilon_greedy :
    def __init__(self, q_values, epsilon):
        self.epsilon = epsilon
        self.q_values = q_values
        self.count = [0] * len(q_values)
        self.avg_reward = 0

    def max_index(self, x):
        m = max(x)
        indexes = [i for i, j in enumerate(x) if j == m]
        return random.choice(indexes)

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.max_index(self.q_values)
        return random.randrange(len(self.q_values))

    def update(self, chosen_arm, reward):
        self.count[chosen_arm] = self.count[chosen_arm] + 1

        self.q_values[chosen_arm] = self.q_values[chosen_arm] + (1/float(self.count[chosen_arm])) * (reward - self.q_values[chosen_arm])
        self.avg_reward = (self.avg_reward * float(sum(self.count) - 1) + reward) / float(sum(self.count))
        return


def iterate_once(algorithm) :
    arm = algorithm.select_arm()
    if arm == 0:
        reward = arm_1_reward()
    if arm == 1:
        reward = arm_2_reward()
    if arm == 2:
        reward = arm_3_reward()
    algorithm.update(arm, reward)


def start_algorithm(number_of_iterates, input_epsilon):
    initial_q = [0] * 3
    initial_epsilon = input_epsilon

    my_algorithm = epsilon_greedy(initial_q, initial_epsilon)

    for i in range(number_of_iterates):
        iterate_once(my_algorithm)
    return my_algorithm.avg_reward, my_algorithm.q_values

each_k_iterate = 1000
k = 10000
epsilon = np.linspace(0, 1, 11)

for j in epsilon:
    result = [0] * 3
    avg_reward_epsilon = 0
    for m in range(1, each_k_iterate):
        avg_reward_once, once_result = start_algorithm(k, j)
        result = [(a*(m-1)+b)/m for a,b in zip(result, once_result)]
        avg_reward_epsilon = (avg_reward_epsilon * float(m - 1) + avg_reward_once) / float(m)
    print("for epsilon {} results average reward is {} and q-values are {}".format(j, avg_reward_epsilon, result))
