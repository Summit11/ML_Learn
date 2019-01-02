from math import log
import operator

def calcShannonEnt(dataset):                        #计算信息熵
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):           #根据特征以及取值划分数据子集
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):            #选择最优划分特征
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)         #计算原始熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):                  #遍历所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                #得到该特征所有取值
        newEntropy = 0.0
        for value in uniqueVals:                  #遍历所有特征所有取值
            subDataSet = splitDataSet(dataSet, i, value)  #根据特征取值划分子集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) #计算条件熵
        infoGain = baseEntropy - newEntropy       #得到信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):                #计算叶子结点所属类别（取叶子节点下样本所属类别最多的）
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):                              #递归构建决策树
    classLists = [example[-1] for example in dataSet]        #获得所有类别
    if classLists.count(classLists[0]) == len(classLists):   #只有数据集一类，返回
        return classLists[0]
    if len(dataSet[0]) == 1:                                #特征已经划分完毕，返回叶子节点类别
        return majorityCnt(classLists)
    bestFeat = chooseBestFeatureToSplit(dataSet)             #得到最优划分特征
    bestFeatLabel = labels[bestFeat]                         #获取特征名
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)                           #最优特征的所有取值
    for value in featValues:                                #递归构建树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataSet = [[1, 1, 'yes'],   #前两列为对应的特征取值，最后一列为对应类别
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']   #特征名
    print(createTree(dataSet, labels))