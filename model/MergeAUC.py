import pandas as pd
from sklearn import metrics
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    pos_label = True

    file_path = '../data/result.xlsx'
    df = pd.read_excel(file_path)
    cn = 'Actual'
    cn2 = 'MCCNN predicted'
    y_true_mccnn = df[cn].values
    y_pred_mccnn = df[cn2].values

    fpr_mccnn, tpr_mccnn, thresholds_mccnn = metrics.roc_curve(y_true_mccnn, y_pred_mccnn, pos_label=1)
    roc_auc_mccnn = metrics.auc(fpr_mccnn, tpr_mccnn)

    file_path = '../data/result.xlsx'
    df = pd.read_excel(file_path)
    cn3 = 'Actual'
    cn4 = 'MetalPrognosis'
    y_true_mp = df[cn3].values
    y_pred_mp = df[cn4].values

    fpr_mp, tpr_mp, thresholds_mp = metrics.roc_curve(y_true_mp, y_pred_mp,pos_label=1)
    roc_auc_mp = metrics.auc(fpr_mp, tpr_mp)

    file_path = '../data/wox_predict.xlsx'
    df = pd.read_excel(file_path)
    cn5 = 'True'
    cn6 = 'Predict'
    y_true_final = df[cn5].values
    y_pred_final = df[cn6].values

    fpr_final, tpr_final, thresholds_final = metrics.roc_curve(y_true_final, y_pred_final, pos_label=1)
    roc_auc_final = metrics.auc(fpr_final, tpr_final)

    file_path = '../data/full_predict.xlsx'
    df = pd.read_excel(file_path)
    cn9 = 'True'
    cn10 = 'Predict'
    y_true2_final = df[cn9].values
    y_pred2_final = df[cn10].values

    fpr2_final, tpr2_final, thresholds2_final = metrics.roc_curve(y_true2_final, y_pred2_final, pos_label=1)
    roc_auc2_final = metrics.auc(fpr2_final, tpr2_final)

    file_path = '../data/Zno_predict.xlsx'
    df = pd.read_excel(file_path)
    cn13 = 'True'
    cn14 = 'Predict'
    y_true3_final = df[cn13].values
    y_pred3_final = df[cn14].values

    fpr3_final, tpr3_final, thresholds4_final = metrics.roc_curve(y_true3_final, y_pred3_final, pos_label=1)
    roc_auc3_final = metrics.auc(fpr3_final, tpr3_final)

    plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = 'Arial'

    plt.plot(fpr_final, tpr_final, color='red', label='MetalTrans ROC curve (AUC = %0.3f)' % roc_auc_final)
    plt.plot(fpr2_final, tpr2_final, color='orange', label='MetalTrans_Each ROC curve (AUC = %0.3f)' % roc_auc2_final)
    plt.plot(fpr_mp, tpr_mp, color='deepskyblue', label='MetalPrognosis_Final ROC curve (AUC = %0.3f)' % roc_auc_mp)

    # plt.plot(fpr_mean, tpr_mean, color='red', label='MetalTrans ROC curve (AUC = %0.3f)' % roc_auc3_final)
    # plt.plot(fpr_mccnn, tpr_mccnn, color='deepskyblue', label='MCCNN ROC curve (AUC = %0.3f)' % roc_auc_mccnn)

    plt.plot([0, 1], [0, 1], 'darkgray', linestyle='--')
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel('False Positive Rate',fontsize=19)
    plt.ylabel('True Positive Rate',fontsize=19)
    plt.title('Multiple ROC Curves',fontsize=20)
    plt.legend(loc='lower right',fontsize=13)
    plt.savefig('../pic/MTP_MergeCurve.png', dpi=300)
    # plt.savefig('../pic/MTM_MergeCurve.png', dpi=600)
    plt.show()
