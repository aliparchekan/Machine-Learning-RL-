from MapBuilder import MapBuilder
import random

class Q_Learning:
    def __init__(self):
        self.q = {}

        self.epsilon = 0.1
        self.gamma = 1
        self.count = {}

    def min_index(self, x):
        m = min(x)
        indexes = [i for i, j in enumerate(x) if j == m]
        return random.choice(indexes)

    def getQs(self, state, action):
        return self.q.get((state, action), 0.0)

    def get_count(self, state, action):
        return self.count.get((state, action), 0)

    def state_min_q(self, mapbuilder, state):
        available = mapbuilder.next_State(state)
        values = [self.getQs(state, a) for a in available]
        return min(values)

    def generate_next(self, mapbuilder, current):
        available_states = mapbuilder.next_State(current)
        action_values = [self.getQs(current, a) for a in available_states]
        if random.random() > self.epsilon:
            chosen = available_states[self.min_index(action_values)]
        else:
            chosen = available_states[random.randrange(len(action_values))]

        return chosen, chosen

    def update_Q(self, current_state, action, reward, next_state, min_value, flag):
        oldvalue = self.getQs(current_state, action)
        count = self.get_count(current_state, action)
        if flag == 0:
            next_value = min_value
        else:
            next_value = 0
        if count == 0:
            self.q[(current_state, action)] = reward + self.gamma * next_value
            self.count[(current_state, action)] = 1
        else:
            self.q[(current_state, action)] = oldvalue + (1/float(count + 1)) * (reward + self.gamma * next_value  - oldvalue)
            self.count[(current_state, action)] = self.count[(current_state, action)] + 1



    def repeat_episode(self, mapbuilder):
        current_state = mapbuilder.initial_state()
        next_state, current_action = self.generate_next(mapbuilder, current_state)
        while current_state != mapbuilder.terminal_state():
            flag = 0
            if next_state == mapbuilder.terminal_state():
                flag = 1
            else:
                min_value_next = self.state_min_q(mapbuilder, next_state)
                two_next_state, next_action = self.generate_next(mapbuilder, next_state)

            fuel, time = mapbuilder.get_Reward(current_state, next_state)
            reward = fuel + time * time

            self.update_Q(current_state, current_action, reward,
                          next_state, min_value_next, flag)
            current_state = next_state
            next_state = two_next_state
            current_action = next_action

    def best_way(self, mapbuilder):
        actions=[]
        for i in range(1,11):
            available = mapbuilder.next_State(i)
            action_values = [self.getQs(i, a) for a in available]
            actions.append(available[self.min_index(action_values)])
            print("for {} values are {}".format(i, action_values))

        return actions




def start(k):
    map = MapBuilder()
    algorithm = Q_Learning()
    for i in range(1,k):
        algorithm.repeat_episode(map)

    print(algorithm.best_way(map))

start(10000)