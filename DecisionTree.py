import numpy as np
from collections import Counter
import random
import copy

from TreeVisualization import createPlot


# 加载数据  150朵鸢尾花,4种特征，3种类别(Iris-setosa, Iris-versicolor, Iris-virginica)
def load_preprocess_data():
    iris_data = []
    # 得到全部数据
    with open("data/iris.data") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split(',')
            for i, e in enumerate(tmp):
                if i < 4:
                    tmp[i] = float(e)
            iris_data.append(tmp)
    print("数据集：\n", iris_data)
    # labels: sepal length（花萼长度） sepal width（花萼宽度） petal length（花瓣长度）, petal width（花瓣宽度）；单位:cm
    iris_labels = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]

    # 四种特征全为连续型数值，将其转换为离散型特征
    # 先找出每列特征的最大最小值
    feat_min = []
    feat_max = []
    for i in range(len(iris_data[0]) - 1):
        feat_min.append(min([row[i] for row in iris_data]))
        feat_max.append(max([row[i] for row in iris_data]))
    print("\n特征最大最小值：")
    print("feat_max", feat_max)
    print("feat_min", feat_min)
    # 每个特征设定一个阈值，小于该阈值特征置为0，否则为1，根据每个特征的最大最小值，将该特征区间等分成十份
    # 尝试每一个节点作为阈值时，计算其划分的数据集信息熵的大小，选择熵最小的节点作为最终划分阈值
    threshold = []
    for i in range(len(feat_min)):
        best_entropy = cal_shannon_entropy(iris_data)
        thre_list = np.linspace(feat_min[i], feat_max[i], 11)
        best_thre = thre_list[0]
        for thre in thre_list:
            t_iris = copy.deepcopy(iris_data)
            discrete_dataset(t_iris, i, thre)
            sub_dataset0 = split_dataset(t_iris, i, 0)
            sub_dataset1 = split_dataset(t_iris, i, 1)
            entropy = len(sub_dataset0) / float(len(t_iris)) * cal_shannon_entropy(sub_dataset0) + len(sub_dataset1) / float(len(t_iris)) * cal_shannon_entropy(sub_dataset1)
            if entropy < best_entropy:
                best_thre = thre
                best_entropy = entropy
        threshold.append(best_thre)
    print("\n离散化特征阈值threshold确定:\n", threshold)
    #根据得到的阈值将数据集离散化
    for i, thre in enumerate(threshold):
        discrete_dataset(iris_data, i, thre)
    return iris_data, iris_labels, threshold

# 根据阈值将数据集特征离散化
def discrete_dataset(dataset, axis, threshold):
    for i in range(len(dataset)):
        if dataset[i][axis] < threshold:
            dataset[i][axis] = 0
        else:
            dataset[i][axis] = 1

# 划分训练集（80%），测试集（20%)
def get_train_test_dadaset(iris_data, split = 0.8):
    num = len(iris_data)
    train_data = []
    test_data = []
    for i in range(num):
        if i % 5 == 0:
            test_data.append(iris_data[i])
        else:
            train_data.append(iris_data[i])
    return train_data, test_data
    # random.shuffle(iris_data) #随机打乱数组顺序
    # return iris_data[:int(num * split)], iris_data[int(num * split):]


# 计算数据集香农信息熵
def cal_shannon_entropy(dataset):
    num = len(dataset)
    label_dic = {}
    for row in dataset:
        if row[-1] not in label_dic.keys():
            label_dic[row[-1]] = 0
        label_dic[row[-1]] += 1

    entropy = 0
    for key in label_dic:
        p = label_dic[key] / float(num)
        entropy -= p * np.log2(p)
    return entropy


# 返回第i个特征值等于value的子数据集，并把该特征列去掉
def split_dataset(dataset, i, value):
    subdateset = []
    for row in dataset:
        if row[i] == value:
            t = []
            t.extend(row[0 : i])
            t.extend((row[i + 1 :]))
            subdateset.append(t)
    return subdateset


# 按照id3/c4.5算法选择最优特征
def choose_feature_to_split(dataset, c45 = False):
    length = len(dataset)
    num_feature = len(dataset[0]) - 1   #特征个数
    base_entropy = cal_shannon_entropy(dataset)
    best_info_gain = 0
    best_info_gain_ratio = 0
    best_feature_index = -1
    for feature_index in range(num_feature):
        feat_list = [row[feature_index] for row in dataset]
        feat_dic = Counter(feat_list) #统计特征种类及出现次数
        cur_entropy = 0
        cur_feat_entropy = 0 # 数据集关于当前特征的信息熵, 用于c4.5算法计算
        for feat in feat_dic:
            sub_dataset = split_dataset(dataset, feature_index, feat)
            sub_entropy = cal_shannon_entropy(sub_dataset)
            cur_entropy += feat_dic[feat] / float(length) * sub_entropy
            cur_feat_entropy -= feat_dic[feat] / float(length) * np.log2(feat_dic[feat] / float(length))

        # 按照c4.5算法规则选择特征
        if c45 == True:
            gain_ration = (base_entropy - cur_entropy) / cur_feat_entropy
            if gain_ration > best_info_gain_ratio:
                best_info_gain_ratio = gain_ration
                best_feature_index = feature_index
        else:
            # 按照id3算法，如果当前特征信息增益最大，那么选择它为最优子特征
            if base_entropy - cur_entropy > best_info_gain:
                best_info_gain = base_entropy - cur_entropy
                best_feature_index = feature_index
    return best_feature_index


# 深度优先构建决策树
def create_decision_tree(dataset, labels, c45 = False):
    class_list = [row[-1] for row in dataset]
    # 如果数据集全部是同一个类别，直接返回该类别
    if len(class_list) == class_list.count(class_list[0]):
        return class_list[0]
    # 如果没有特征可以继续划分或者特征区分度为0，返回出现次数最多的类别
    if len(dataset[0]) == 1 or choose_feature_to_split(dataset, c45) == -1:
        class_dic = Counter(class_list)
        return class_dic.most_common()[0][0]

    # 根据id3/c4.5算法选择信息增益最高的特征作为当前分类特征
    best_feature_index = choose_feature_to_split(dataset, c45)    #将c45改为True即使用c4.5规则选择分类特征
    best_feature_label = labels[best_feature_index]

    labels.remove(best_feature_label)
    # 以字典的形式构建决策树
    decision_tree = {best_feature_label : {}}
    feat_list = [row[best_feature_index] for row in dataset]
    feat_dic = Counter(feat_list)
    # 根据当前最优特征划分子数据集
    for feat in feat_dic:
        sub_dataset = split_dataset(dataset, best_feature_index, feat)
        sub_labels = labels[:]
        # 递归构建子数据集决策树
        decision_tree[best_feature_label][feat] = create_decision_tree(sub_dataset, sub_labels)
    return decision_tree

# 根据决策树对数据进行预测
def predict(tree, labels_index, input):
    for key in tree:
        if isinstance(tree[key][input[labels_index[key]]], str):
            return tree[key][input[labels_index[key]]]
        else:
            return predict(tree[key][input[labels_index[key]]], labels_index, input)


# 将决策树可视化


if __name__ == "__main__":
    dataset, labels, threshold = load_preprocess_data()
    train_data, test_data = get_train_test_dadaset(dataset)
    for t in test_data:
        print(t)
    # 构建决策树，用字典形式保存
    decision_tree = create_decision_tree(train_data, labels, c45 = True)
    print("\n决策树字典形式表示：\n", decision_tree)

    #测试集预测，计算准确率
    labels_index = {"花萼长度" : 0, "花萼宽度" : 1, "花瓣长度" : 2, "花瓣宽度" : 3}
    num = len(test_data)
    correct_num = 0
    for t in test_data:
        if predict(decision_tree, labels_index, t) == t[-1]:
            correct_num += 1
        else:
            print(t, " wrong answer:", predict(decision_tree, labels_index, t))

    print("\n预测准确率 : {:.2%}".format(float(correct_num) / num))
    createPlot(decision_tree)






