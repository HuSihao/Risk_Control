#-*- coding:utf-8 -*-
import igraph as ig
import numpy as np
import pandas as pd


def main():

    Data = pd.read_csv('./data/graph_data.csv')
    Data[[u'l_user_phone']].values()


    g = ig.Graph()




if __name__ == '__main__':

    main()