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

# relMobile
X = data[data['relMobile'].notnull()]
X = X[X['relMobile']!='d41d8cd98f00b204e9800998ecf8427e']['relMobile']
dic = {}
for i in X.index:
    try:
        dic[X.loc[i]].append(i)
    except:
        dic[X.loc[i]] = []
        dic[X.loc[i]].append(i)

for L in dic.values(): #dic.values()是list的集合
    if len(L)>1:#说明存在边
        for i in xrange(len(L)):
            for j in xrange(i+1,len(L)):
                g.add_edges([(L[i],L[j])])

# workplace
X = data[data['workplace'].notnull()]
X1 = X[X['workplace'] != '个体']
X2 = X1[X1['workplace']!='自由职业']
X3 = X2[X2['workplace']!='个体户']
X = X3[X3['workplace']!='无']['workplace']
dic = {}
for i in X.index:
    try:
        dic[X.loc[i]].append(i)
    except:
        dic[X.loc[i]] = []
        dic[X.loc[i]].append(i)

for L in dic.values(): #dic.values()是list的集合
    if len(L)>1:#说明存在边
        for i in xrange(len(L)):
            for j in xrange(i+1,len(L)):
                g.add_edges([(L[i],L[j])])

#'order_mac'
X1 = data[data['order_mac'].notnull()]
X = X1[X1['order_mac']!='未知']['order_mac']
dic = {}
for i in X.index:
    try:
        dic[X.loc[i]].append(i)
    except:
        dic[X.loc[i]] = []
        dic[X.loc[i]].append(i)

for L in dic.values(): #dic.values()是list的集合
    if len(L)>1:#说明存在边
        for i in xrange(len(L)):
            for j in xrange(i+1,len(L)):
                g.add_edges([(L[i],L[j])])

#全部属性建图
for attribute in attributes:
    print attribute,
    X = data[data[attribute].notnull()][attribute]
    dic = {}
    for i in X.index:
        try:
            dic[X.loc[i]].append(i)
        except:
            dic[X.loc[i]] = []
            dic[X.loc[i]].append(i)
    #一个attribute的dic完成
    for L in dic.values(): #dic.values()是list的集合
        if len(L)>1:#说明存在边
            for i in xrange(len(L)):
                for j in xrange(i+1,len(L)):
                    g.add_edges([(L[i],L[j])])
    print 'graph edge #:',g.ecount()
