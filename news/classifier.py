# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import interp
from itertools import cycle

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

from keras.layers import Dense, Input, Conv1D, Dropout, Embedding, MaxPooling1D, Flatten, Concatenate
from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
from keras.initializers import Constant

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn import metrics as sm
from cm_clr import plot_classification_report, plot_confusion_matrix

# Any results you write to the current directory are saved as output.
df = pd.read_json('dataset.json', lines=True)
df.head()

df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
df.category = df.category.map(lambda x: "ARTS & CULTURE" if x == "ARTS" else x)
df.category = df.category.map(lambda x: "ARTS & CULTURE" if x == "CULTURE & ARTS" else x)
df.category = df.category.map(lambda x: "STYLE & BEAUTY" if x == "STYLE" else x)
df.category = df.category.map(lambda x: "HEALTHY LIVING" if x == "WELLNESS" else x)
df.category = df.category.map(lambda x: "SCIENCE & TECH" if x == "SCIENCE" else x)
df.category = df.category.map(lambda x: "SCIENCE & TECH" if x == "TECH" else x)

cates = df.groupby('category')
print(cates.size())

# using headlines and short_description as input X
df['text'] = df.headline + " " + df.short_description

# tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X

# delete some empty and short data
df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

df.head()

df.word_length.describe()

# using 50 for padding length
maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))

# category to id
categories = df.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k:i})
    int_category.update({i:k})

df['c2id'] = df['category'].apply(lambda x: category_int[x])

# glove embedding
word_index = tokenizer.word_index

EMBEDDING_DIM = 100

embeddings_index = {}
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s unique tokens.' % len(word_index))
print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False)

# prepared data 
X = np.array(X)
Y = np_utils.to_categorical(list(df.c2id))

# and split to training set and validation set
seed = 29
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)

inp = Input(shape=(maxlen,), dtype='int32')
embedding = embedding_layer(inp)
stacks = []
for kernel_size in [2, 3, 4]:
    conv = Conv1D(64, kernel_size, padding='same', activation='relu', strides=1)(embedding)
    pool = MaxPooling1D(pool_size=3)(conv)
    drop = Dropout(0.5)(pool)
    stacks.append(drop)

merged = Concatenate()(stacks)
flatten = Flatten()(merged)
drop = Dropout(0.5)(flatten)
outp = Dense(len(int_category), activation='softmax')(drop)

TextCNN = Model(inputs=inp, outputs=outp)
TextCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

TextCNN.summary()

checkpoint = ModelCheckpoint("outputs/model.hdf5", monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger("outputs/history.csv", separator=',', append=False)
tb = TensorBoard("outputs/tb_logs/{}".format(time.time()))
#python -m tensorboard.main --logdir=PATH/TO/TB_Logs/
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=4, verbose=1, mode='max', min_lr=0.0001)
    
textcnn_history = TextCNN.fit(x_train, 
                              y_train, 
                              batch_size=128, 
                              epochs=20, 
                              validation_data=(x_val, y_val), 
                              callbacks=[checkpoint, csv_logger, tb, reduce_lr])

model = load_model("outputs/model.hdf5")

y_pred = model.predict(x_val)
predictions, actuals = [], []
for i in range(len(y_pred)): 
    predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
    actuals.append(np.where(y_val[i] == np.max(y_val[i]))[0][0])
    
n_classes = len(categories)

cm = sm.confusion_matrix(predictions, actuals, labels=[i for i in range(n_classes)])
plot_confusion_matrix(cm, classes=categories, title='Confusion Matrix')
plt.savefig("outputs/cnf_mtx.png", dpi=200, format='png', bbox_inches='tight'); plt.close();

for i in range(len(categories)): 
    categories[i] = categories[i].replace("&", "and")
    categories[i] = categories[i].replace(" ", "_")

cr = sm.classification_report(actuals, predictions, labels=[i for i in range(n_classes)], target_names=categories); cr = cr.split("\n")
plot_classification_report(cr[0] + '\n\n' + 
                           cr[2] + '\n' + cr[3] + '\n' + cr[4] + '\n' + cr[5] + '\n' + cr[6] + '\n' + 
                           cr[7] + '\n' + cr[8] + '\n' + cr[9] + '\n' + cr[10] + '\n' + cr[11] + '\n' + 
                           cr[12] + '\n' + cr[13] + '\n' + cr[14] + '\n' + cr[15] + '\n' + cr[16] + '\n' + 
                           cr[17] + '\n' + cr[18] + '\n' + cr[19] + '\n' + cr[20] + '\n' + cr[21] + '\n' + 
                           cr[22] + '\n' + cr[23] + '\n' + cr[24] + '\n' + cr[25] + '\n' + cr[26] + '\n' + 
                           cr[27] + '\n' + cr[28] + '\n' + cr[29] + '\n' + cr[30] + '\n' + cr[31] + '\n' + 
                           cr[32] + '\n' + cr[33] + '\n' + cr[34] + '\n' + cr[35] + '\n' + cr[36] + '\n' + 
                           cr[38] + '\n', title = 'Classification Report')
plt.savefig("outputs/cls_rep.png", dpi=200, format='png', bbox_inches='tight'); plt.close();

pd.Series([ y[0] for y in y_pred ]).plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
plt.xlabel('Values'); plt.ylabel('Frequency')
plt.title('Predictions (y) Ranges Histogram')
plt.grid(axis='y', alpha=0.75)
plt.savefig("outputs/pred_dist.png", dpi=200, format='png', bbox_inches='tight'); plt.close();

lw = 2

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = sm.roc_curve(y_val[:, i], y_pred[:, i])
    roc_auc[i] = sm.auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = sm.roc_curve(y_val.ravel(), y_pred.ravel())
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
plt.savefig("outputs/roc_auc.png", dpi=200, format='png', bbox_inches='tight'); plt.close();

auc = sm.roc_auc_score(y_val, y_pred)
y_pred = [ np.argmax(y) for y in y_pred ]; y_val = [ np.argmax(y) for y in y_val ]
acc = sm.accuracy_score(y_val, y_pred)
k = sm.cohen_kappa_score(y_val, y_pred)

f = open("outputs/Metrics.txt","w+")
f.write("\n::::::::::::::::: OVERALL OUTPUT DERIVATIONS :::::::::::::::::")
f.write('\n\nAccuracy: ' + str(acc) + '%' +
      '\n\nCohen\'s Kappa Co-efficient (K): ' + str(k) +
      '\n\nArea Under the Curve (AUC): ' + str(auc) + '\n')
f.close()