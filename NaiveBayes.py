import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):              #创建词集
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def bagOfWords2Vec(vocabList, inputSet):   #词袋模型
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word : %s is not in my Vocabulary' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)        #文档数
    numWords = len(trainMatrix[0])         #总词数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  #垃圾文档概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords) #在两类文档下，每个词出现次数
    p0Denonm = 2.0; p1Denom = 2.0                        #在两类文档下，出现的总词数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                        #垃圾文档
            p1Num += trainMatrix[i]                      #累加每个词的个数
            p1Denom += sum(trainMatrix[i])               #累加总词数
        else:
            p0Num += trainMatrix[i]
            p0Denonm += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)                     #在垃圾文档条件下，每个词出现的概率
    p0Vect = np.log(p0Num / p0Denonm)                    #在非垃圾文档条件下，每个词出现的概率
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)       #计算当前文档的后验概率
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():                                #测试
    listOPosts, listClasses = loadDataSet()     #获取数据集
    myVocabList = createVocabList(listOPosts)   #得到词集
    trainMat = []                               #词向量矩阵
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)  #得到条件概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))  #分类
    print(testEntry,' classified as : ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
   testingNB()