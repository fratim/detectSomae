import numpy as np
import dataIO
from sklearn.metrics import confusion_matrix



def getMetrics(true_labels, pred_labels):

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    print ("TP: {}, FP: {}, TN: {}, FN: {}".format(TP,FP,TN,FN))

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    ACC = (TP+TN)/(TP+TN+FP+FN)

    print("TPR: ")
    print(TPR)
    print("FPR: ")
    print(FPR)
    print("Acc: ")
    print(ACC)


somae_gt = dataIO.ReadH5File("Mouse/gt_data/somae_Mouse_762x832x832.h5",[1])
somae_pred = dataIO.ReadH5File("Mouse/gt_data/somae_pred_Mouse.h5",[1])
somae_pred_refined = dataIO.ReadH5File("Mouse/gt_data/JWR-somae_refined-dsp_8.h5",[1])

somae_gt = somae_gt[384:,:,:]
somae_gt[somae_gt!=0]=1

somae_pred_refined[somae_pred_refined!=0]=1

print("Metrics for Network output: ")
getMetrics(somae_gt, somae_pred)

print("Metrics for refined: ")
getMetrics(somae_gt, somae_pred_refined)
