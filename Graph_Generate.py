#-*- coding:utf-8 -*-
import numpy as np
from igraph import *
import pandas as pd
import time
data = pd.read_csv('../data/jimi/jimi_data/graph_data.csv')
attributes = ['userId','l_user_phone','order_ip','order_mac','GPS','GPS_100','GPS_1000','order_ldeviceId'] #可以直接使用的attribute
# workplace,relMobile需要经过处理 有频率很大的数据，会生成很多无用边

# Initial Graph
g = Graph()
g.add_vertices(data.shape[0])

def main():

    Data = pd.read_csv('./data/graph_data.csv')
    Data[[u'l_user_phone']].values()


    g = ig.Graph()




if __name__ == '__main__':

    main()