import numpy as np
import random
import bigfloat

movement_reward = -10
key_reward = 100
police_reward = -150
finish_reward = 200
camera_reward = -80

class prison:
    def __init__(self):
        self.maze = np.array([
            [1., 0., 1., 1., 1.],
            [0., 1., 0., 1., 1.],
            [1., 1., 1., 1., 0.],
            [1., 0., 1., 1., 1.],
            [1., 0., 1., 1., 1.],
            [1., 1., 1., 1., 1.]
        ])

        self.nrows, self.ncolumns = self.maze.shape
        self.key_target1 = [1, 3]
        self.key_target2 = [5, 3]
        self.final_target = [1, 1]
        self.first_q = {}
        self.second_q = {}
        self.first_e = {}
        self.second_e = {}
        self.count1 = {}
        self.count2 = {}
        self.first_episode = []
        self.second_episode = []
        self.gamma = 0.9
        self.landa = 0.9
        self.policy_number = 0
        self.epsilon1 = 0.5
        self.epsilon2 = 0.5
        self.reset()


    def reset(self):
        self.position = [5, 1]
        self.first_e.clear()
        self.second_e.clear()
        self.first_episode[:] = []
        self.second_episode[:] = []
        self.policy_number = 0


    def update_state(self, action):
        next_state = [self.position[0], self.position[1]]
        if action == 0: # left
            next_state[1] = self.position[1] - 1
        elif action == 1: # up
            next_state[0] = self.position[0] - 1
        elif action == 2: # right
            next_state[1] = self.position[1] + 1
        else: # down
            next_state[0] = self.position[0] + 1
        return next_state


    def police_position(self):
        a = random.random()
        b = random.random()
        if a < 0.5:
            first = 4
        else:
            first = 5
        if b < 0.5:
            second = 3
        else:
            second = 4

        return [first, second]

    def get_reward(self,state, number_of_policy):
        if state == self.police_position():
            return police_reward
        if self.maze[state[0]][state[1]] == 0:
            return camera_reward

        if number_of_policy == 0:
            if state == self.key_target1:
                return key_reward
            if state == self.key_target2:
                return key_reward
        else:
            if state == self.final_target:
                return finish_reward
        return movement_reward

    def getQ1(self,state,action):
        return self.first_q.get((state,action), 0)

    def getQ2(self, state, action):
        return self.second_q.get((state, action), 0)

    def gete1(self, state, action):
        return self.first_e.get((state, action), 0)

    def gete2(self, state, action):
        return self.second_e.get((state, action), 0)

    def getcount1(self,state,action):
        return self.count1.get((state,action), 0)

    def getcount2(self,state,action):
        return self.count2.get((state,action), 0)

    def max_index(self, x):
        m = max(x)
        indexes = [i for i, j in enumerate(x) if j == m]
        return random.choice(indexes)

    def choose_action(self,state, actions, number_of_policy):
        if number_of_policy == 0:
            values = [self.getQ1(state, a) for a in actions]
            if random.random() > self.epsilon1:
                chosen = self.max_index(values)
            else:
                chosen = random.randrange(len(values))

        else:
            values = [self.getQ2(state, a) for a in actions]
            if random.random() > self.epsilon2:
                chosen = self.max_index(values)
            else:
                chosen = random.randrange(len(values))

        return actions[chosen]

    def valid_actions(self,state):
        actions = [0, 1, 2, 3]
        row = state[0]
        column = state[1]

        if row == 0:
            actions.remove(1)
        elif row == self.nrows - 1:
            actions.remove(3)

        if column == 0:
            actions.remove(0)
        elif column == self.ncolumns - 1:
            actions.remove(2)

        if state == [0,3]:
            actions.remove(2)
        if state == [1,2]:
            actions.remove(3)
        if state == [1,3]:
            actions.remove(3)
        if state == [1,4]:
            actions.remove(1)
        if state == [2,2]:
            actions.remove(1)
        if state == [2,3]:
            actions.remove(1)
        if state == [3,3]:
            actions.remove(3)
        if state == [3,4]:
            actions.remove(3)
        if state == [4,2]:
            actions.remove(2)
        if state == [4,3]:
            actions.remove(0)
            actions.remove(1)
        if state == [4,4]:
            actions.remove(1)

        return actions

    def identify_end(self, current_reward, state):
        if current_reward == police_reward:
            self.reset()
            return False
        if self.policy_number == 0:
            if state == self.key_target2:
                self.policy_number = 1
                return True
            elif state == self.key_target1:
                self.policy_number = 1
                return True
        else:
            if state == self.final_target:
                self.reset()
                return False


    def update_q(self, number_of_policy, state, action, delta):
        if number_of_policy == 0:
            old_value = self.getQ1(state,action)
            count = self.getcount1(state,action)
            if count == 0:
                self.first_q[(state, action)] = delta * self.gete1(state,action)
                self.count1[(state, action)] = 1
            else:
                self.first_q[(state, action)] = old_value + (1/float(count + 1)) * delta * self.gete1(state, action)
                self.count1[(state,action)] = count + 1
        else:
            old_value = self.getQ2(state, action)
            count = self.getcount2(state, action)
            if count == 0:
                self.second_q[(state, action)] = delta * self.gete2(state, action)
                self.count2[(state, action)] = 1
            else:
                self.second_q[(state, action)] = old_value + (1 / float(count + 1)) * delta * self.gete2(state, action)
                self.count2[(state, action)] = count + 1

    def update_es(self, number_of_policy):
        if number_of_policy == 0:
            for k in self.first_e:
                self.first_e[k] = self.first_e[k] * self.landa * self.gamma
        else:
            for k in self.second_e:
                self.second_e[k] = self.second_e[k] * self.landa * self.gamma
    def updating(self, number_of_policy, delta):
        if number_of_policy == 0:
            for k in self.first_episode:
                state = [k[0][0], k[0][1]]
                action = k[1]
                self.update_q(number_of_policy,(state[0],state[1]), action, delta)
            self.update_es(number_of_policy)
        else:
            for k in self.second_episode:
                state = [k[0][0], k[0][1]]
                action = k[1]
                self.update_q(number_of_policy,(state[0],state[1]), action, delta)
            self.update_es(number_of_policy)


    def act(self):
        valid_actions = self.valid_actions(self.position)
        current_action = self.choose_action((self.position[0],self.position[1]), valid_actions, 0)
        flag = True

        while flag == True:
            next_state = self.update_state(current_action)
            next_valid_actions = self.valid_actions(next_state)
            next_action = self.choose_action((next_state[0],next_state[1]), next_valid_actions, self.policy_number)
            current_reward = self.get_reward(next_state, self.policy_number)
            if self.policy_number == 0:
                delta = current_reward + self.gamma * self.getQ1((next_state[0],next_state[1]),next_action) - self.getQ1((self.position[0], self.position[1]),current_action)
                self.first_e[((self.position[0],self.position[1]), current_action)] = 1
                t = ((self.position[0], self.position[1]), current_action)
                self.first_episode.append(t)
            else:
                delta = current_reward + self.gamma * self.getQ2((next_state[0], next_state[1]),
                                                                 next_action) - self.getQ2(
                    (self.position[0], self.position[1]), current_action)
                self.second_e[((self.position[0], self.position[1]), current_action)] = 1
                t = ((self.position[0], self.position[1]), current_action)
                self.second_episode.append(t)
            self.updating(self.policy_number, delta)
            change = self.policy_number
            flag = self.identify_end(current_reward, self.position)
            if change != self.policy_number:
                continue
            self.position = next_state
            current_action = next_action

    def show_result(self):
        result1 = np.zeros((6,5))
        result2 = np.zeros((6,5))
        rv1 = np.zeros((6,5))
        rv2 = np.zeros((6,5))
        for i in range(6):
            for j in range(5):
                valid_actions = self.valid_actions([i,j])
                values1 = [self.getQ1((i,j), a) for a in valid_actions]
                values2 = [self.getQ2((i,j), a) for a in valid_actions]
                result1[i][j] = valid_actions[values1.index(max(values1))]
                result2[i][j] = valid_actions[values2.index(max(values2))]
                rv1[i][j] = max(values1)
                rv2[i][j] = max(values2)

        print(rv1)
        print('')
        print(rv2)

        return result1, result2



def start(number):
    my_prison = prison()
    for i in range(1,number):
        my_prison.act()

    res1 , res2 = my_prison.show_result()
    print(res1)
    print("")
    print(res2)

start(20000)
