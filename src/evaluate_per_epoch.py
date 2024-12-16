from sklearn import metrics
import datetime
import os
import sys
import yaml

# 准确率
def calculate_accuracy(predictions, labels):
    total_samples = len(predictions)
    correct_samples = sum([1 for pred, label in zip(predictions, labels) if pred == label])
    accuracy = (correct_samples / total_samples) * 100
    return accuracy

# 精确率
def calculate_precision(predictions, labels):
    TP = sum([1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1])
    FP = sum([1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0])
    precision = TP / (TP + FP)
    return precision

# 召回率
def calculate_recall(predictions, labels):
    TP = sum([1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1])
    FN = sum([1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1])
    recall = TP / (TP + FN)
    return recall

# F1
def calculate_f1(predictions, labels):
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# AUC
def calculate_auc(predictions, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

# 输入一个list，保存至文件txt中，文件名为当前系统时间+“epoch”+epoch数.txt，文件中每行是一个list中的元素
def save_list_to_file(list1,list2, epoch, log_dir):
    current_time = datetime.datetime.now()
    log_file = os.path.join(log_dir, current_time.strftime("%Y-%m-%d-%H-%M-%S") + 'epoch' + str(epoch) + '.txt')
    with open(log_file, 'w') as f:
        for i  in range(len(list1)):
            # 每行写入list1[i]，空格和list2[i]
            f.write(str(list1[i]) + ' ' + str(list2[i]) + '\n')


# 计算准确率，精确率，召回率，F1，AUC，将结果保存在log文件中，log文件命名用系统时间，存入指定文件夹。
def evaluate_and_log_per_epoch(predictions_tmp, labels_tmp):
    # 将predictions_tmp和labels_tmp中的元素转换为int
    predictions = []
    labels = []
    for i in predictions_tmp:
        if i > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    for i in labels_tmp:
        if i > 0.5:
            labels.append(1)
        else:
            labels.append(0)
    accuracy = calculate_accuracy(predictions, labels)
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    f1 = calculate_f1(predictions, labels)
    auc = calculate_auc(predictions, labels)
    return accuracy, precision, recall, f1, auc
    
