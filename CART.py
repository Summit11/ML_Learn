import numpy as np
import pandas as pd

def create_samples():
    '''''
    提供训练样本集
    每个example由多个特征值+1个分类标签值组成
    比如第一个example=['youth', 'no', 'no', '1', 'refuse'],此样本的含义可以解读为：
    如果一个人的条件是：youth age，no working, no house, 信誉值credit为1
    则此类人会被分类到refuse一类中，即在相亲中被拒绝(也可以理解为银行拒绝为此人贷款)
    每个example的特征值类型为：
    ['age', 'working', 'house', 'credit']
    每个example的分类标签class_label取值范围为：'refuse'或者'agree'
    '''
    datas = [['youth', 'no',  'no',   '1', 'refuse'],
                 ['youth', 'no',  'no',   '2', 'refuse'],
                 ['youth', 'yes', 'no',   '2', 'agree'],
                 ['youth', 'yes', 'yes',  '1', 'agree'],
                 ['youth', 'no',  'no',   '1', 'refuse'],
                 ['mid',   'no',  'no',   '1', 'refuse'],
                 ['mid',   'no',  'no',   '2', 'refuse'],
                 ['mid',   'yes', 'yes',  '2', 'agree'],
                 ['mid',   'no',  'yes',  '3', 'agree'],
                 ['mid',   'no',  'yes',  '3', 'agree'],
                 ['elder', 'no',  'yes',  '3', 'agree'],
                 ['elder', 'no',  'yes',  '2', 'agree'],
                 ['elder', 'yes', 'no',   '2', 'agree'],
                 ['elder', 'yes', 'no',   '3', 'agree'],
                 ['elder', 'no',  'no',   '1', 'refuse']]
    featName = ['age', 'working', 'house', 'credit']
    featList = set(np.arange(0, len(featName)))
    datas = np.array(datas)
    return datas, featList, featName

class CART:
    def __init__(self, datas, featList, featName, gainThre, featNumThre, sampleNumThre):
        self.datas = datas
        self.featList = featList
        self.featName = featName
        self.gainThre = gainThre
        self.featNumThre = featNumThre
        self.sampleNumThre = sampleNumThre

        self.bestfeatIdx = None
        self.bestfeatValue = None
        self.leftTree = None
        self.rightTree = None
        self.result = None

    def calGini(self, datas):
        y_labels = np.unique(datas[:, -1])
        rows = len(datas)
        y_p = {}
        gini = 1.0
        for y_label in y_labels:
            y_p[y_label] = len(datas[datas[:, -1] == y_label]) / rows
            gini -= (y_p[y_label] ** 2)
        return gini

    def getMaxCate(self):
        y_labels = np.unique(self.datas[:, -1])
        y_n = {}
        for y_label in y_labels:
            y_n[y_label] = len(self.datas[self.datas[:, -1] == y_label])
        return max(y_n.items(), key = lambda x: x[1])[0]

    def createTree(self):
        if len(self.featList) <= self.featNumThre or len(self.datas) <= self.sampleNumThre:
            self.result = self.getMaxCate()
        else:
            initGini = self.calGini(self.datas)
            maxGain = 0.0
            maxLeftData = None
            maxRightData = None

            for featIdx in self.featList:
                featValueSet = set(self.datas[:, featIdx])
                for featValue in featValueSet:
                    leftData = self.datas[self.datas[:, featIdx] == featValue]
                    rightData = self.datas[self.datas[:, featIdx] != featValue]
                    gini = len(leftData) / len(self.datas) * self.calGini(leftData) \
                           + len(rightData) / len(self.datas) * self.calGini(rightData)
                    gain = initGini - gini

                    if gain > maxGain:
                        maxGain = gain
                        self.bestfeatIdx = featIdx
                        self.bestfeatValue = featValue
                        maxLeftData = leftData
                        maxRightData = rightData

            if maxGain <= self.gainThre:
                self.result = self.getMaxCate()
            else:
                self.featList.remove(self.bestfeatIdx)
                leftTree = CART(maxLeftData, self.featList, self.featName,self.gainThre, self.featNumThre, self.sampleNumThre)
                leftTree.createTree()
                rightTree = CART(maxRightData, self.featList, self.featName, self.gainThre, self.featNumThre, self.sampleNumThre)
                rightTree.createTree()
                self.leftTree = leftTree
                self.rightTree = rightTree

    def printTree(self):
        if self.leftTree == None or self.rightTree == None:
            return self.result
        else:
            resList = {}
            resList[self.featName[self.bestfeatIdx] + '=' + self.bestfeatValue] = self.leftTree.printTree()
            resList[self.featName[self.bestfeatIdx] + '<>' + self.bestfeatValue] = self.rightTree.printTree()
            return resList



if __name__ == '__main__':
    datas, featList, featName = create_samples()
    Tree = CART(datas, featList, featName, 0, 0, 3)
    Tree.createTree()
    res = Tree.printTree()
    print(res)