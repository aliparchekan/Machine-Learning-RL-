import random
import numpy as np
import math

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


class UCB1:
    def __init__(self, q_values):
        self.q_values = q_values
        self.count = [0] * len(q_values)
        self.avg_reward = 0
        self.R = [0] * len(q_values)


    def max_index(self, x):
        return x.index(max(x))


    def select_arm(self):
        length = len(self.q_values)
        for i in range(length):
            if self.count[i] == 0:
                return i

        greedy_values = [0] * length
        total_k = sum(self.count)
        for i in range(length):
            ucb = math.sqrt(2 * math.log(total_k) / self.count[i]) * self.R[i]
            greedy_values[i] = ucb + self.q_values[i]

        return self.max_index(greedy_values)

    def update(self, chosen_arm, reward):
        self.count[chosen_arm] = self.count[chosen_arm] + 1
        self.q_values[chosen_arm] = self.q_values[chosen_arm] + (1/float(self.count[chosen_arm])) * (reward - self.q_values[chosen_arm])
        self.avg_reward = (self.avg_reward * float(sum(self.count) - 1) + reward) / float(sum(self.count))
        if (abs(self.q_values[chosen_arm]) > self.R[chosen_arm]):
            self.R[chosen_arm] = abs(self.q_values[chosen_arm])


def iterate_once(algorithm):
    arm = algorithm.select_arm()
    if arm == 0:
        reward = arm_1_reward()
    if arm == 1:
        reward = arm_2_reward()
    if arm == 2:
        reward = arm_3_reward()
    algorithm.update(arm, reward)

def start_algorithm(number_of_iterates):
    initial_q = [0] * 3

    my_algorithm = UCB1(initial_q)

    for i in range(number_of_iterates):
        iterate_once(my_algorithm)
    return my_algorithm.avg_reward, my_algorithm.q_values

each_k_iterate = 1000
k = 10000

result = [0] * 3
average_reward = 0
for j in range(1,each_k_iterate):
    avg_reward_once, once_result = start_algorithm(k)
    result = [(a * (j - 1) + b) / j for a, b in zip(result, once_result)]
    average_reward = (average_reward * float(j - 1) + avg_reward_once) / float(j)

print("average reward is {} and q-values are {}".format( average_reward, result))