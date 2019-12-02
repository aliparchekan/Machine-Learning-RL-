import numpy as np
import matplotlib.pyplot as ppl

class Value_Iterate:
    def __init__(self):
        self.rewards = np.zeros(101)
        self.rewards[100] = 1
        self.V = np.zeros(101)
        self.discount = 0.9
        self.p = 0.4
        self.theta = 0.0001


    def action_values(self, s):
        actions = range(1, s + 1)
        action_value = np.zeros(101)

        for i in actions:
            action_value[i] = self.p * (self.rewards[min(100,s + i)] + self.V[min(100,s + i)] * self.discount) + (1 - self.p) * (self.rewards[max(0,s - i)] + self.V[max(s - i, 0)] * self.discount)

        return action_value


    def update(self):
        while True:
            delta = 0
            for s in range(1,100):
                v = self.V[s]
                A = self.action_values(s)
                self.V[s] = max(A)
                delta = max(delta, np.abs(v - self.V[s]))

            if delta < self.theta:
                break




    def return_policy(self):
        policy = np.zeros(100)
        for s in range(1,100):
            A = self.action_values(s)
            policy[s] = np.argmax(A)

        return policy, self.V



algorithm = Value_Iterate()
algorithm.update()

policy, value = algorithm.return_policy()

x = range(100)
y_1 = value[:100]
y_2 = policy

ppl.figure(1)
ppl.plot(x, y_1)

ppl.xlabel('States')
ppl.ylabel('State values')

ppl.title('State Values vs. States')

ppl.figure(2)
ppl.stem(x, y_2)

ppl.xlabel('States')
ppl.ylabel('Policy')

ppl.title('Policy in each State')
print(policy)
print('')
print(value)

ppl.show()




