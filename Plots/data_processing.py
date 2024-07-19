import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc

ones = np.ones(180)
zeros = np.zeros(185)
y_true = np.concatenate((ones, zeros))

def process_prediction(name, name2):
    result = []
    with open(name, 'r') as f:
        for line in f:
            result.append(float(line))

    with open(name2, 'r') as f:
        for line in f:
            result.append(float(line))

    return np.array(result)

plt.clf()
bp = [150, 300, 500, 1000, 3000]
for i in range(len(bp)):
    virus = 'virus_' + str(bp[i]) + '.fa_gt1bp_dvfpred.txt'
    bac = 'bac_' + str(bp[i]) + '.fa_gt1bp_dvfpred.txt'
    y_scores = process_prediction(virus, bac)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    plt.plot(recall, precision, label='bp:' + str(bp[i]) + ', AUPRC:' + str(round(auprc, 5)))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.show()