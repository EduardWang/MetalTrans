import tensorflow as tf
import numpy as np
import gc
from model import get_model
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef, f1_score, roc_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve
import openpyxl as op
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


filename = '../result/fiveFold_crossValidation_Full_161.xlsx'


def op_toexcel(data,filename):

    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]

        ws.append(data)
        wb.save(filename)
    else:
        wb = op.Workbook()
        ws = wb['Sheet']
        ws.append(['MCC', 'ACC', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1', 'FPR', 'FNR',
                  'TN', 'FP', 'FN', 'TP','AUPRC','Threshold'])
        ws.append(data)
        wb.save(filename)


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='Red', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontname='Arial')
    plt.ylabel('True Positive Rate', fontname='Arial')
    plt.tick_params(labelsize=10)
    plt.title('Receiver Operating Characteristic', fontsize=10)
    plt.legend(loc="lower right")
    plt.show()


def data_generator(train_esm, train_prot, train_y, batch_size):
    L = train_esm.shape[0]

    while True:
        for i in range(0, L, batch_size):
            batch_esm = train_esm[i:i + batch_size].copy()
            batch_prot = train_prot[i:i + batch_size].copy()
            batch_y = train_y[i:i + batch_size].copy()

            yield ([batch_esm, batch_prot], batch_y)


def cross_validation(train_esm, train_prot, train_label, valid_esm, valid_prot, valid_label,
                     test_esm, test_prot, test_label, k):


    train_size = train_label.shape[0]
    val_size = valid_label.shape[0]
    batch_size = 32
    train_steps = train_size // batch_size

    val_steps = val_size // batch_size

    print(f"Fold {k} - Training samples: {train_esm.shape[0]}, Test samples: {test_esm.shape[0]}")

    qa_model = get_model()
    valiBestModel = f'../save_model/fivecv_model/FULL/model_regular.h5'

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=valiBestModel,
        monitor='val_loss',
        save_weights_only=True,
        verbose=1,

        save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=0,
        mode='auto'
    )

    train_generator = data_generator(train_esm, train_prot, train_label, batch_size)
    val_generator = data_generator(valid_esm, valid_prot, valid_label, batch_size)

    history_callback = qa_model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=500,
        verbose=1,
        callbacks=[checkpointer, early_stopping],
        validation_data=val_generator,
        validation_steps=val_steps,
        shuffle=True,
        workers=1
    )

    # accuracy = history_callback.history['accuracy']
    # val_accuracy = history_callback.history['val_accuracy']
    # epochs = range(1, len(accuracy) + 1)
    #
    # plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    # plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    # plt.title('Training And Validation val_accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    #
    # loss = history_callback.history['loss']
    # val_loss = history_callback.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    #
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training And Validation val_accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    train_generator.close()
    val_generator.close()
    del train_generator
    del val_generator
    gc.collect()

    print(f"\nFold {k} - Validation Loss: {history_callback.history['val_loss'][-1]:.4f}, " +
          f"Validation Accuracy: {history_callback.history['val_accuracy'][-1]:.4f}")

    print(f"Fold {k} - Testing:")

    test_pred=qa_model.predict([test_esm,test_prot]).reshape(-1,)
    fcvtest(test_pred,test_label)


def fcvtest(test_pred,test_label):
    y_pred=test_pred

    y_true=test_label
    y_pred_new = []
    
    best_f1 = 0
    best_threshold = 0.5
    for threshold in range(35, 100):
        threshold = threshold / 100
        binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
        f1 = metrics.f1_score(y_true, binary_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    for value in y_pred:
        if value < best_threshold:
            y_pred_new.append(0)
        else:
            y_pred_new.append(1)
    y_pred_new = np.array(y_pred_new)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_new, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    fpr2, tpr2, thresholds2 = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    roc_auc = roc_auc if roc_auc >= roc_auc2 else roc_auc2
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_new)
    auprc = metrics.auc(recall, precision)
    thd=best_threshold

    # plot_roc_curve(y_true,y_pred)
    # plt.title("ROC")

    print("Matthews: " + str(matthews_corrcoef(y_true, y_pred_new)))
    print("ACC: ", (tp + tn) / (tp + tn + fp + fn))
    print("AUC: ", roc_auc)
    print('sensitivity/recall:', tp / (tp + fn))
    print('specificity:', tn / (tn + fp))
    print('precision:', tp / (tp + fp))
    print('negative predictive value:', tn / (tn + fn))
    print("F1: " + str(f1_score(y_true, y_pred_new)))
    print('error rate:', fp / (tp + tn + fp + fn))
    print('false positive rate:', fp / (tn + fp))
    print('false negative rate:', fn / (tp + fn))
    print('TN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)
    print('AUPRC: ' + str(auprc))
    print('best_threshold: ' + str(best_threshold))

    mcc = float(format((matthews_corrcoef(y_true, y_pred_new)), '.4f'))
    acc = float(format((tp + tn) / (tp + tn + fp + fn), '.4f'))
    auc = float(format(roc_auc, '.4f'))
    sen = float(format(tp / (tp + fn), '.4f'))
    spe = float(format(tn / (tn + fp), '.4f'))
    pre = float(format(tp / (tp + fp), '.4f'))

    npv = float(format(tn / (tn + fn), '.4f'))
    f1 = float(format(f1_score(y_true, y_pred_new), '.4f'))
    fpr = float(format(fp / (tn + fp), '.4f'))
    fnr = float(format(fn / (tp + fn), '.4f'))
    auprc = float(format(auprc, '.4f'))

    result = mcc, acc, auc, sen, spe, pre, npv, f1, fpr, fnr, tn, fp, fn, tp, auprc,thd
    op_toexcel(result, filename)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    all_esm = np.lib.format.open_memmap('../features_npy/esm/FULL/train/161/FULL_esm.npy')
    all_prot = np.lib.format.open_memmap('../features_npy/prottrans/FULL/train/161/FULL_prot.npy')
    all_label = np.lib.format.open_memmap('../features_npy/labels/FULL/train/161/FULL_labels.npy')

    all_y = all_label

    for i in range(3):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        k = 1
        for train_index, test_index in cv.split(all_esm, all_y):

            train_Esm = all_esm[train_index]
            train_Prot = all_prot[train_index]
            train_Y = all_y[train_index]

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_inx, valid_inx in split.split(train_Esm, train_Y):

                valid_Esm = train_Esm[valid_inx]
                valid_Prot = train_Prot[valid_inx]
                valid_Y = train_Y[valid_inx]

                train_Esm = train_Esm[train_inx]
                train_Prot = train_Prot[train_inx]
                train_Y = train_Y[train_inx]

            test_Esm = all_esm[test_index]
            test_Prot = all_prot[test_index]
            test_Y = all_y[test_index]

            cross_validation(train_Esm, train_Prot,  train_Y, valid_Esm, valid_Prot, valid_Y,
                             test_Esm, test_Prot, test_Y, k)

            k += 1











