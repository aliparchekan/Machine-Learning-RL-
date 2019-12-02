import random
import numpy as np
import math

p = 0.5
q = 0.6
l = 0.6


def arm_1_reward():
    prob = random.random()
    if prob <= p :
        r = np.random.normal(60, 8)
    else:
        r = np.random.normal(-40, 8)
    if r > 0:
        return 1
    return 0


def arm_2_reward():
    prob = random.random()
    if prob <= q:
        r = np.random.uniform(40, 60)
    else:
        r = np.random.uniform(-70, -40)
    if r > 0:
        return 1
    return 0

def arm_3_reward():
    prob = random.random()
    if prob <= l:
        r = np.random.normal(20, 8)
    else:
        r = np.random.uniform(-10, 10)
    if r > 0:
        return 1
    return 0


class TS:
    def __init__(self, len):
        self.S = [0] * len
        self.F = [0] * len

    def max_index(self, x):
        return x.index(max(x))

    def select_arm(self):
        length = len(self.S)
        theta = [0] * length

        for i in range(length):
            theta[i] = np.random.beta(self.S[i] + 1, self.F[i] + 1)

        return self.max_index(theta)


    def update(self, chosen_arm, reward):
        if reward == 1:
            self.S[chosen_arm] = self.S[chosen_arm] + 1
        else:
            self.F[chosen_arm] = self.F[chosen_arm] + 1

    def give_expectation(self, index):
        return np.random.beta(self.S[index] + 1, self.F[index] + 1)

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
    my_algorithm = TS(3)

    for i in range(number_of_iterates):
        iterate_once(my_algorithm)
    result_a =[]
    result_a.append(my_algorithm.give_expectation(0))
    result_a.append(my_algorithm.give_expectation(1))
    result_a.append(my_algorithm.give_expectation(2))
    return result_a

k = 10000
each_k_iterate = 1000


result = [0] * 3
for j in range(1,each_k_iterate):
    once_result = start_algorithm(k)
    result = [(a * (j - 1) + b) / j for a, b in zip(result, once_result)]

print("average prize won from first arm is {} ".format(result[0] * 1000))
print("average prize won from second arm is {} ".format(result[1] * 1000))
print("average prize won from third arm is {} ".format(result[2] * 1000))