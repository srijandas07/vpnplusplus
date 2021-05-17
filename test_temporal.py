from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from _name_to_int import _name_to_int
import pandas as pd
import numpy as np
import os

labels = pd.read_csv("test_Labels.csv")
y_pred = np.load('pred_arr.txt.npy')


ground_truth = open("/data/stars/user/sdas/smarthomes_data/splits/test_new_CS.txt", "r")
lines = ground_truth.readlines()
names = [os.path.splitext(i.strip())[0] for i in lines]
y_true = [_name_to_int(os.path.splitext(i.strip())[0].split('_')[0], 'CS') - 1 for i in lines]

accuracies = list()
gcnt = 0
pcnt = 0

while pcnt<len(y_pred):
    acc = list()
    while (pcnt<len(y_pred) and labels['name'][pcnt]==names[gcnt]):
        acc.append(y_pred[pcnt].tolist())
        pcnt += 1
    accuracies.append(np.argmax(np.sum(acc, axis=0), -1))
    gcnt += 1

print(len(accuracies))
print(accuracy_score(y_true, accuracies))
print(balanced_accuracy_score(y_true, accuracies))
cnf = confusion_matrix(y_true, accuracies)
np.save("cnf.txt", cnf)
