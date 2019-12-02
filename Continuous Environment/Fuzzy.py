import gym
import numpy as np
import random

class fuzzy:
    def __init__(self, position_label, velocity_label, actions):
        self.env = gym.make("MountainCarContinuous-v0")
        self.position_label = position_label
        self.velocity_label = velocity_label
        self.alpha = 0.0035
        self.rules = []
        self.actions = actions
        self.sigma_position = 0.01
        self.sigma_velocity = 0.001
        self.epsilon = 1
        self.gama = 0.9
        self.flag = 0
        for i in self.position_label:
            for j in self.velocity_label:
                self.rules.append((i, j))

        self.q_value = {}



    def get_q(self, position_l, velocity_l, action):
        return self.q_value.get((position_l, velocity_l, action), 0)

    def membership_return(self, x, c, s):
        a = np.exp(-((x - c)**2)/s)
        return a

    def find_rules_min_membership(self, rule, state):
        a1 = self.membership_return(state[0], rule[0], self.sigma_position)
        a2 = self.membership_return(state[1], rule[1], self.sigma_velocity)
        return min(a1, a2)

    def choose_rule_action(self, rule):
        values = [self.get_q(rule[0], rule[1], x) for x in self.actions]
        if random.random() > self.epsilon:
            chosen = self.max_index(values)
        else:
            chosen = random.randrange(len(values))

        return self.actions[chosen]

    def find_firing_rates(self, state):
        rates = []
        for i in self.rules:
            rates.append(self.find_rules_min_membership(i, state))
        return rates

    def find_all_rules_actions(self):
        chosens = []
        for i in self.rules:
            chosens.append(self.choose_rule_action(i))
        return chosens

    def find_rule_max_action_value(self, rule):
        values = [self.get_q(rule[0], rule[1], x) for x in self.actions]
        return max(values)

    def find_all_rules_max_actions_values(self):
        chosens = []
        for i in self.rules:
            chosens.append(self.find_rule_max_action_value(i))
        return chosens


    def choose_continuous_action(self, state):
        actions = self.find_all_rules_actions()
        rates = self.find_firing_rates(state)
        value = 0
        for i in range(len(rates)):
            value = value + rates[i] * actions[i]

        return value / sum(rates)

    def calculate_state_max_value(self, state):
        rates = self.find_firing_rates(state)
        max_values = self.find_all_rules_max_actions_values()
        result = 0
        for i in range(len(rates)):
            result = result + rates[i] * max_values[i]
        return result / sum(rates)

    def calculate_state_chosen_value(self, state):
        rates = self.find_firing_rates(state)
        actions = self.find_all_rules_actions()
        values = []
        result = 0
        for i in range(len(self.rules)):
            values.append(self.get_q(self.rules[i][0], self.rules[i][1], actions[i]))

        for i in range(len(rates)):
            result = result + rates[i] * values[i]
        return result / sum(rates)

    def update_a_chosen_q(self, rule, action, delta, state, rule_number):
        value = self.get_q(rule[0], rule[1], action)
        rates = self.find_firing_rates(state)
        value = value + self.alpha * (rates[rule_number] / sum(rates)) * delta
        self.q_value[(rule[0], rule[1], action)] = value

    def update_all_qs(self, delta, state):
        actions = self.find_all_rules_actions()
        for i in range(len(self.rules)):
            self.update_a_chosen_q(self.rules[i], actions[i], delta, state, i)

        return

    def calculate_delta(self, reward, state, next_state):
        max_next = self.calculate_state_max_value(next_state)
        chosen = self.calculate_state_chosen_value(state)

        return reward + self.gama * max_next - chosen

    def update(self):
        state = self.env.reset()
        chosen_action = self.choose_continuous_action(state)
        #self.env.render()
        c = 0
        while state[0] < 0.5:
            c = c + 1
            if (c % 1000) == 0:
                print(c)
            observation, reward, done, info = self.env.step(np.array([chosen_action]))
            #self.env.render()
            if observation[0] > 0.49:
                reward = 100
            delta = self.calculate_delta(reward, state, observation)
            self.update_all_qs(delta, state)

            next_action = self.choose_continuous_action(observation)
            state = observation
            chosen_action = next_action
            if self.flag == 1:
                self.epsilon = self.epsilon* 0.9999
                if self.epsilon < 0.1:
                    self.epsilon = 0.1

        print("it converged in {} steps".format(c))
        self.flag = 1

    def max_index(self, x):
        m = max(x)
        indexes = [i for i, j in enumerate(x) if j == m]
        return random.choice(indexes)


position_label = [-1, -0.5, 0 , 0.3]
velocity_label = [-0.05, -0.02, 0, 0.02, 0.05]
actions = [-1, 0 , 1]
def start(number):
    algo = fuzzy(position_label,velocity_label, actions)
    for i in range(number):
        algo.update()








start(3)