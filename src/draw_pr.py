import os
import matplotlib.pyplot as plt
nocall_csv_path = r'/home/dj/gnn2/evaluate/new/nocall_PR.csv'
gnn2_csv_path = r'/home/dj/gnn2/evaluate/new/GNN2_PR.csv'
safe_csv_path = r'/home/dj/gnn2/evaluate/new/safe_PR.csv'



def draw_precision_recall_curve(file_path):
    """
    Draw Precision-Recall curve based on the input file.
    """
    import csv
    predictions = []
    labels = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        for row in reader:
            prediction, label = row[0], row[1]
            predictions.append(float(prediction))
            labels.append(float(label))
    return labels, predictions

class Curve:
    def __init__(self, path, name):
        self.labels, self.predictions = draw_precision_recall_curve(path)
        self.name = name
        self.precision, self.recall = self.get_precision_recall()


    def draw_precision_recall_curve(self, file_path):
        """
        Draw Precision-Recall curve based on the input file.
        """
        import csv
        predictions = []
        labels = []

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            # Skip the header
            next(reader)
            for row in reader:
                prediction, label = row[0], row[1]
                predictions.append(float(prediction))
                labels.append(float(label))
        return labels, predictions
    
    def get_precision_recall(self):
        """
        Get precision and recall.
        """
        import numpy as np
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(self.labels, self.predictions)
        return precision, recall
    
    # 删除recall为0的点
    def del_recall0(self, list1,list2):
        list3 = []
        list4 = []
        for i in range(len(list1)):
            if list1[i] <= 0.01:
                continue
            else:
                list3.append(list1[i])
                list4.append(list2[i])
        return list3, list4



curve1 = Curve(nocall_csv_path, 'nocall')
curve2 = Curve(gnn2_csv_path, 'gnn2')
curve3 = Curve(safe_csv_path, 'safe')
#plt.plot(curve1.recall, curve1.precision, label=curve1.name)
plt.plot(curve2.recall, curve2.precision, label=curve2.name)
plt.plot(curve3.recall, curve3.precision, label=curve3.name)
#plt.plot(curve4.recall, curve4.precision, label=curve4.name)

curve1_new_recall, curve1_new_precision = curve1.del_recall0(curve1.recall, curve1.precision)
plt.plot(curve1_new_recall, curve1_new_precision, label=curve1.name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

output_path = r'/home/dj/gnn2/evaluate/new/PR.png'
#检查文件是否已存在，存在则删除
if os.path.exists(output_path):
    print('remove file: ', output_path)
    os.remove(output_path)
    print('removed')
plt.savefig(output_path)
print('saved')