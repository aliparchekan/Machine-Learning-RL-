import networkx as nx
import numpy as np
import pandas as pd


class MapBuilder:

    def __init__(self):
        self.data_df = pd.read_csv('data.csv')
        number_of_edges = len(self.data_df)
        self.G = nx.DiGraph()

        for i in range(0, number_of_edges):
            start = self.data_df.iloc[i]['start']
            end = self.data_df.iloc[i]['end']
            Energy = self.data_df.iloc[i]['Energy']
            Time = self.data_df.iloc[i]['Time']

            self.G.add_edge(start, end, Energy=Energy, Time=Time)

    def next_State(self, state):
        if (state == 11):
            print('This state is terminal state and has no successor!')
        edges = [n for n in self.G.neighbors(state)]
        return edges

    def get_Reward(self, start, end):
        edge = self.data_df.loc[(self.data_df['start'] == (start)) & (self.data_df['end'] == (end))]
        Energy = edge.iloc[0]['Energy']
        Time = edge.iloc[0]['Time']
        return [np.random.normal(Energy, 0.5), np.random.normal(Time, 0.5)]

    def initial_state(self):
        return 1

    def terminal_state(self):
        return 11