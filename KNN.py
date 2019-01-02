'''
Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
Output:     the most popular class label
'''
import numpy as np
import operator

def NormData(dataSet):               #数据归一化
    minVals = dataSet.min(axis = 0)  #每列最小值，对应某个特征的最小值
    maxvals = dataSet.max(axis = 0)  #每列最大值，对应某个特征的最大值
    ranges = maxvals - minVals
    retData = (dataSet - minVals) / ranges #归一化
    return retData

def KNNClassify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                     #样本数
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet  #目标样本和每个样本求差值
    sqDiffMat = diffMat**2                             #差值平方
    sqDistances = sqDiffMat.sum(axis=1)                #按行求平方和
    distances = sqDistances**0.5                       #开根号得到距离
    sortedDistIndicies = distances.argsort()           #排序
    classCount={}                                      #前k个样本，{类别:个数}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]                      #返回前k个最多的类别


if __name__ == '__main__':
    dataSet = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    lables = ['A', 'B', 'A', 'B']
    iX = [0.5, 0.5, 0.5]
    print(KNNClassify(iX, dataSet, lables, 2))