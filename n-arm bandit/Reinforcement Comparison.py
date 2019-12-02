import csv
import numpy as np
import bigfloat
from matplotlib import pyplot as ppl


a = 0.4
b = 0.6
P = 2

channels_click = [[[] for i in range(3)] for i in range(2)]
channels_view = [[[] for i in range(3)] for i in range(2)]
channels_cost = [[[] for i in range(3)] for i in range(2)]


def purchase(number_of_views, number_of_clicks):
    return a * number_of_views + b * number_of_clicks


def Benefit(purchases, cost):
    return purchases * P - cost


with open('Q4.csv') as file:
    read = csv.reader(file, delimiter = ',')
    for row in read:
        channels_view[0][0].append(row[0])
        channels_view[0][1].append(row[1])
        channels_view[0][2].append(row[2])
        channels_click[0][0].append(row[3])
        channels_click[0][1].append(row[4])
        channels_click[0][2].append(row[5])
        channels_cost[0][0].append(row[6])
        channels_cost[0][1].append(row[7])
        channels_cost[0][2].append(row[8])
        channels_view[1][0].append(row[9])
        channels_view[1][1].append(row[10])
        channels_view[1][2].append(row[11])
        channels_click[1][0].append(row[12])
        channels_click[1][1].append(row[13])
        channels_click[1][2].append(row[14])
        channels_cost[1][0].append(row[15])
        channels_cost[1][1].append(row[16])
        channels_cost[1][2].append(row[17])
for i in range(2):
    for j in range(3):
        for k in range(3):
            del channels_view[i][j][0]
            del channels_click[i][j][0]
            del channels_cost[i][j][0]
        channels_view[i][j] = [float(x) for x in channels_view[i][j]]
        channels_cost[i][j] = [float(x) for x in channels_cost[i][j]]
        channels_click[i][j] = [float(x) for x in channels_click[i][j]]


class RC:
    def __init__(self, q_initial):
        self.q_values = q_initial
        self.avg_reward = 0
        self.action_count = [1] * len(q_initial)
        self.count = 0
        self.beta = 0.9
        self.alpha = 0.1
        self.benefit = 0
        self.theta = 1
        self.true_avg_reward = [0]
        self.true_q_values = [0] * len(q_initial)

    def select_action(self):
        preference = [bigfloat.exp(x/self.theta, bigfloat.precision(50)) for x in self.q_values]
        sumation = sum(preference)
        prob = [float(x)/float(sumation) for x in preference]
        arg = np.random.choice(len(self.q_values), 1, prob)
        return arg[0]

    def update(self, chosen_arm, benefit):
        self.count = self.count + 1
        self.q_values[chosen_arm] = self.q_values[chosen_arm] + self.beta * (benefit - self.avg_reward)
        self.avg_reward = self.avg_reward + self.alpha * (benefit - self.avg_reward)
        self.benefit = (float(self.benefit) * float(self.count - 1) + benefit ) / float(self.count)
        self.action_count[chosen_arm] = self.action_count[chosen_arm] + 1
        temp = self.true_avg_reward[-1] * len(self.true_avg_reward)
        temp = temp + benefit
        self.true_avg_reward.append(temp / (len(self.true_avg_reward) + 1))
        self.true_q_values[chosen_arm] = (self.true_q_values[chosen_arm] * (self.action_count[chosen_arm] - 1) + benefit) / self.action_count[chosen_arm]
        self.theta = self.theta * 0.99


def iterate_once(algorithm, day):
    action = algorithm.select_action()
    if action == 0:
        view = channels_view[0][0][day]
        click = channels_click[0][0][day]
        cost = channels_cost[0][0][day]
    if action == 1:
        view = channels_view[0][1][day]
        click = channels_click[0][1][day]
        cost = channels_cost[0][1][day]
    if action == 2:
        view = channels_view[0][2][day]
        click = channels_click[0][2][day]
        cost = channels_cost[0][2][day]
    if action == 3:
        view = channels_view[1][0][day]
        click = channels_click[1][0][day]
        cost = channels_cost[1][0][day]
    if action == 4:
        view = channels_view[1][1][day]
        click = channels_click[1][1][day]
        cost = channels_cost[1][1][day]
    if action == 5:
        view = channels_view[1][2][day]
        click = channels_click[1][2][day]
        cost = channels_cost[1][2][day]

    num_of_purchase = purchase(view, click)
    benefit = Benefit(num_of_purchase, cost)
    algorithm.update(action, benefit)


def start_algorithm(number_of_iterate):
    initial_q = [0] * 6

    my_algorithm = RC(initial_q)

    for i in range(number_of_iterate):
        iterate_once(my_algorithm, i)

    true_q = my_algorithm.true_q_values
    avg = my_algorithm.true_avg_reward


    return my_algorithm.avg_reward,my_algorithm.q_values , avg , true_q

each_iterate = 100
k = len(channels_cost[0][0])
result = 0
q_action = [0] * 6
avg_action = [0] * 6
avg_regret = [0] * k

result = 0
q_action = [0] * 6
for i in range(1, each_iterate):
    algorithm_avg_benefit, action_q , avg , true_q= start_algorithm(k)
    result = (float(result) * float(i - 1) + algorithm_avg_benefit) / float(i)
    q_action = [(a*(i - 1) + b)/i for a,b in zip(q_action, action_q)]
    regret = [max(true_q) - x for x in avg]
    del regret[0]
    avg_regret = [(a * (i - 1) + b)/ i for a,b in zip(avg_regret, regret)]

print("The average reward obtained is {}".format(result))
print("The benefit for channel 1 in time 0-8 is {}".format(true_q[0]))
print("The benefit for channel 1 in time 8-16 is {}".format(true_q[1]))
print("The benefit for channel 1 in time 16-24 is {}".format(true_q[2]))
print("The benefit for channel 2 in time 0-8 is {}".format(true_q[3]))
print("The benefit for channel 2 in time 8-16 is {}".format(true_q[4]))
print("The benefit for channel 2 in time 16-24 is {}".format(true_q[5]))

ppl.plot(range(20,k + 1), avg_regret[19:])
ppl.show()







