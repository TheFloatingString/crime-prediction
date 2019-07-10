
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import os
from sklearn.model_selection import train_test_split as TTS
from scipy import interp
from itertools import cycle
from sklearn import metrics as sm
from cm_clr import plot_classification_report, plot_confusion_matrix
from keras.models import load_model, Sequential
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
import time


# In[6]:

weather_data = np.load("preprocessed_data/X_train_weather.npy")
nasdaq_data = np.load("preprocessed_data/X_train_nasdaq.npy")
print(weather_data.shape, nasdaq_data.shape)


# In[3]:


base = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")
end = datetime.datetime.strptime("2019-07-06", "%Y-%m-%d")
delta = end - base
unique_PI_dates = [str(base + datetime.timedelta(days=x)).split(' ')[0] for x in range(delta.days + 1)]
print(len(unique_PI_dates))


# In[7]:


pdq_list = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 
            10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 
            20.0, 21.0, 22.0, 23.0, 24.0, 26.0, 
            27.0, 30.0, 31.0, 33.0, 35.0, 38.0, 
            39.0, 42.0, 44.0, 45.0, 46.0, 48.0, 
            49.0, 50.0, 55.0]

for pdq in pdq_list:
    
    try: os.makedirs("models/" + str(pdq))
    except FileExistsError: pass
    
    pdq_filename = "pdq_" + str(int(pdq)) + '.csv'
    
    pdq_df = pd.read_csv("datasets/" + pdq_filename, index_col='Unnamed: 0')
    target_df = np.split(pdq_df, [6], axis=1)[0]
    daytime_df = np.split(pdq_df, [6], axis=1)[1]
    
    y = target_df.to_numpy()
    X = np.concatenate((weather_data, nasdaq_data, daytime_df.to_numpy()), axis=1)
    
    print(X.shape, y.shape)
    
    X_train, X_test, y_train, y_test = TTS(X, y, test_size = 0.1)

    #class_weights = compute_class_weight('balanced', list(np.unique(y_train).flatten()), list(y_train.flatten()))
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    classifier = Sequential()
    
    classifier.add(Conv1D(128, X_train.shape[1], activation='relu', input_shape=(X_train.shape[1], 1)))
    classifier.add(MaxPooling1D(pool_size = (1), strides=(10)))
    classifier.add(Conv1D(64, 1, strides=1, padding='valid', activation="relu", kernel_initializer='glorot_uniform'))
    classifier.add(MaxPooling1D(pool_size = (1), strides=(10)))
    classifier.add(Conv1D(32, 1, strides=1, padding='valid', activation="relu", kernel_initializer='glorot_uniform'))
    classifier.add(MaxPooling1D(pool_size = (1), strides=(10)))
    
    classifier.add(Dropout(0.166))
    classifier.add(Flatten())
    classifier.add(Dropout(0.166))
    
    classifier.add(Dense(activation="relu", units=128, kernel_initializer="uniform"))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation="relu", units=64, kernel_initializer="uniform"))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation="relu", units=32, kernel_initializer="uniform"))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation="relu", units=4, kernel_initializer="uniform"))
    classifier.add(BatchNormalization())
    classifier.add(Dense(activation="softmax", units=6, kernel_initializer="uniform"))
    
    classifier.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    checkpoint = ModelCheckpoint("models/"+str(pdq)+"/"+"model.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger("models/"+str(pdq)+"/"+"history.csv", separator=',', append=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.0001)
    
    history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 10, 
                             callbacks=[checkpoint, csv_logger, reduce_lr], epochs = 100)
    
    model = load_model("models/"+str(pdq)+"/"+"model.hdf5")
    
    y_pred = model.predict(X_test)
    predictions = [ np.argmax(y) for y in y_pred ]; actuals = [ np.argmax(y) for y in y_test ]
        
    cl = ["Introduction", "Vol dans / sur véhicule à moteur", "Vol de véhicule à moteur", "Méfait", "Vols qualifiés", "Infractions entrainant la mort"]
    n_classes = len(cl)
    
    cm = sm.confusion_matrix(predictions, actuals, labels=[i for i in range(n_classes)])
    plot_confusion_matrix(cm, classes=cl, title='Confusion Matrix')
    plt.savefig("models/"+str(pdq)+"/"+"cnf_mtx.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    cr = sm.classification_report(actuals, predictions, labels=[i for i in range(n_classes)], target_names=cl); cr = cr.split("\n")
    plot_classification_report(cr[0] + '\n\n' + cr[2] + '\n' + cr[3] + '\n' + cr[4] + '\n' + cr[5] + '\n' + cr[6] + '\n' + cr[7] + '\n' + cr[11] + '\n', title = 'Classification Report')
    plt.savefig("models/"+str(pdq)+"/"+"cls_rep.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    pd.Series([ y[0] for y in y_pred ]).plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.xlabel('Values'); plt.ylabel('Frequency')
    plt.title('Predictions (y) Ranges Histogram')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("models/"+str(pdq)+"/"+"pred_dist.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    lw = 2
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sm.roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = sm.auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = sm.roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = sm.auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes): mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr; tpr["macro"] = mean_tpr
    roc_auc["macro"] = sm.auc(fpr["macro"], tpr["macro"])
    
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig("models/"+str(pdq)+"/"+"roc_auc.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    fig = plt.figure(figsize=(24, 16))
    
    fig.suptitle('\n\n\n\n\n', fontsize=25)
    
    fig.add_subplot(211); plt.imshow(plt.imread("models/"+str(pdq)+"/"+"cls_rep.png")); plt.axis('off'); os.remove("models/"+str(pdq)+"/"+"cls_rep.png")
    fig.add_subplot(234); plt.imshow(plt.imread("models/"+str(pdq)+"/"+"cnf_mtx.png")); plt.axis('off'); os.remove("models/"+str(pdq)+"/"+"cnf_mtx.png")
    fig.add_subplot(212); plt.imshow(plt.imread("models/"+str(pdq)+"/"+"roc_auc.png")); plt.axis('off'); os.remove("models/"+str(pdq)+"/"+"roc_auc.png")
    fig.add_subplot(236); plt.imshow(plt.imread("models/"+str(pdq)+"/"+"pred_dist.png")); plt.axis('off'); os.remove("models/"+str(pdq)+"/"+"pred_dist.png")
    plt.savefig("models/"+str(pdq)+"/"+"output_derivations.png", dpi=600, format='png'); plt.close();
