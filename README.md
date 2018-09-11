# Jimi Risk Control Model
## 1.对用于建图的数据进行Homophilic Test
见Homophilic Test.ipynb

|Relation |Edge|CrossEdge|CrossEdgeFraction|Connectedness|Dyadicity |Heterophilicity |
|---------|----|---------|-----------------|-------------|----------|----------------|
|User Id  |  25135  |  0  |   0   |   |   |
|User phone| 25135  |  0  |   0   |   |   | 
|order ip |  9753   | 579 | 0.059 |   |   |    
|GPS      |  18358  | 393 | 0.021 |   |   |    
|GPS_100  |	89046  |3634 | 0.041 |   |   |    
|GPS_1000 |  377510 |21272| 0.056 |   |   |    
|workplace|  17594 | 860  | 0.049 |   |   |    
|order mac|  76402 | 6024 | 0.079 |   |   |    
|order ldeviceId | 1155 | 10 | 0.009 |    |    
|relMobile|  2008  |  9   | 0.004 |   |   |    

## 2.关系网络构建

### 2.1 构建Unipartite-Graph
见Unipartite Graph.ipynb
### 2.2 构建Bipartite-Graph

## 3.图特征提取
### 3.1 Graph-Embedding
### 3.2 Bipartite-Graph 统计特征抽取

## 4.有监督模型分类
