import matplotlib.pyplot as plt

def getNumLeafs(tree):
    numLeafs = 0
    #获取第一个节点的分类特征
    firstFeat = list(tree.keys())[0]
    #得到firstFeat特征下的决策树（以字典方式表示）
    secondDict = tree[firstFeat]
    #遍历firstFeat下的每个节点
    for key in secondDict.keys():
        #如果节点类型为字典，说明该节点下仍然是一棵树，此时递归调用getNumLeafs
        if type(secondDict[key]).__name__== 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        #否则该节点为叶节点
        else:
            numLeafs += 1
    return numLeafs

#获取决策树深度
def getTreeDepth(tree):
    maxDepth = 0
    #获取第一个节点分类特征
    firstFeat = list(tree.keys())[0]
    #得到firstFeat特征下的决策树（以字典方式表示）
    secondDict = tree[firstFeat]
    #遍历firstFeat下的每个节点，返回子树中的最大深度
    for key in secondDict.keys():
        #如果节点类型为字典，说明该节点下仍然是一棵树，此时递归调用getTreeDepth，获取该子树深度
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
#画出决策树

def createPlot(tree):
    # 定义一块画布，背景为白色
    fig = plt.figure(1, facecolor='white')
    # 清空画布
    fig.clf()
    # 不显示x、y轴刻度
    xyticks = dict(xticks=[], yticks=[])
    # frameon：是否绘制坐标轴矩形
    createPlot.pTree = plt.subplot(111, frameon=False, **xyticks)
    # 计算决策树叶子节点个数
    plotTree.totalW = float(getNumLeafs(tree))
    # 计算决策树深度
    plotTree.totalD = float(getTreeDepth(tree))
    # 最近绘制的叶子节点的x坐标
    plotTree.xOff = -0.5 / plotTree.totalW
    # 当前绘制的深度：y坐标
    plotTree.yOff = 1.0
    # （0.5,1.0）为根节点坐标
    plotTree(tree, (0.5, 1.0), '')
    plt.show()

# 定义决策节点以及叶子节点属性：boxstyle表示文本框类型，sawtooth：锯齿形；fc表示边框线粗细
decisionNode = dict(boxstyle="sawtooth", fc="0.5")
leafNode = dict(boxstyle="round4", fc="0.5")

# 定义箭头属性
arrow_args = dict(arrowstyle="<-")


# nodeText:要显示的文本；centerPt：文本中心点，即箭头所在的点；parentPt：指向文本的点；nodeType:节点属性
# ha='center'，va='center':水平、垂直方向中心对齐；bbox：方框属性
# arrowprops：箭头属性
# xycoords，textcoords选择坐标系；axes fraction-->0,0是轴域左下角，1,1是右上角
def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.pTree.annotate(nodeText, xy=parentPt, xycoords="axes fraction",
                              xytext=centerPt, textcoords='axes fraction',
                              va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def plotMidText(centerPt, parentPt, midText):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]
    createPlot.pTree.text(xMid, yMid, midText)

def plotTree(tree, parentPt, nodeTxt):
    #计算叶子节点个数
    numLeafs = getNumLeafs(tree)
    #获取第一个节点特征
    firstFeat = list(tree.keys())[0]
    #计算当前节点的x坐标
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    #绘制当前节点
    plotMidText(centerPt,parentPt,nodeTxt)
    plotNode(firstFeat,centerPt,parentPt,decisionNode)
    secondDict = tree[firstFeat]
    #计算绘制深度
    plotTree.yOff -= 1.0/plotTree.totalD
    for key in secondDict.keys():
        #如果当前节点的子节点不是叶子节点，则递归
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],centerPt,str(key))
        #如果当前节点的子节点是叶子节点，则绘制该叶节点
        else:
            #plotTree.xOff在绘制叶节点坐标的时候才会发生改变
            plotTree.xOff += 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,plotTree.yOff),centerPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),centerPt,str(key))
    plotTree.yOff += 1.0/plotTree.totalD
