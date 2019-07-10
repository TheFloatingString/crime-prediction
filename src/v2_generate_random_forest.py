import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn import metrics as sm
from cm_clr import plot_classification_report, plot_confusion_matrix
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

from math import sqrt


pdq_list = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 20.0, 21.0, 22.0, 23.0, 24.0, 26.0, 27.0, 30.0, 31.0, 33.0, 35.0, 38.0, 39.0, 42.0, 44.0, 45.0, 46.0, 48.0, 49.0, 50.0, 55.0]

for pdq_number in pdq_list:
<<<<<<< HEAD

	print("pdq_number: "+str(pdq_number))
	df_unique_crime_pdq = pd.read_csv("datasets/pdq_"+str(int(pdq_number))+".csv", index_col="Unnamed: 0").drop(columns=["jour", "nuit", "soir"])
	# print(df_unique_crime_pdq.head())
	# df_nasdaq = pd.read_csv("datasets/NASDAQ_DATA.csv", index_col="Date")
	# print(df_nasdaq.head())
	nasdaq_data = np.load("preprocessed_data/X_train_nasdaq.npy")
	# print(nasdaq_data[0])
	weather_array = np.load("preprocessed_data/X_train_weather.npy")

	# x[~np.isnan(x).any(axis=1)]
	# x[x == -inf] = 0

	X_data = np.concatenate((nasdaq_data, weather_array), axis=1)
	y_data = df_unique_crime_pdq.values

	X_data[X_data == np.inf] = 0
	X_data[X_data == -np.inf] = 0

	# scaling data
	X_min_max_scaler = MinMaxScaler()
	X_data = X_min_max_scaler.fit_transform(X_data)
	joblib.dump(X_min_max_scaler, "static/saved_models/X_min_max_scaler.pkl")

	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

	# print(weather_array[0])
	clf = RandomForestRegressor()
	clf.fit(X_train, y_train)
	joblib.dump(clf, "static/saved_models/v2_rn_forest_"+str(pdq_number)+".pkl")
	print(clf.score(X_test, y_test))

	# mse
	predictions = clf.predict(X_test)
	mse = sqrt(mean_squared_error(predictions, y_test))
	print("mse: "+str(mse))

	# accuracy
	correct = 0
	for y_pred, y in zip(predictions, y_test):
		pred_answer = list(y_pred).index(max(y_pred))
		actual_answer = list(y).index(max(y))
		if pred_answer == actual_answer:
			correct += 1

	accuracy = correct/len(y_test)
	print("acc: "+str(accuracy))

	random_test_index = random.randrange(0, len(X_test))
	print(random_test_index)

	print(clf.predict([X_test[random_test_index]]))
	print(y_test[random_test_index])
=======
    
    print("pdq_number: "+str(pdq_number))
    df_unique_crime_pdq = pd.read_csv("../datasets/pdq_"+str(int(pdq_number))+".csv", index_col="Unnamed: 0").drop(columns=["jour", "nuit", "soir"])
    print(df_unique_crime_pdq.head())
    # df_nasdaq = pd.read_csv("datasets/NASDAQ_DATA.csv", index_col="Date")
    # print(df_nasdaq.head())
    nasdaq_data = np.load("../preprocessed_data/X_train_nasdaq.npy")
    # print(nasdaq_data[0])
    weather_array = np.load("../preprocessed_data/X_train_weather.npy")
    
    # x[~np.isnan(x).any(axis=1)]
    # x[x == -inf] = 0
    
    X_data = np.concatenate((nasdaq_data, weather_array), axis=1)
    y_data = df_unique_crime_pdq.values
    
    X_data[X_data == np.inf] = 0
    X_data[X_data == -np.inf] = 0
    
    # scaling data
    X_min_max_scaler = MinMaxScaler()
    X_data = X_min_max_scaler.fit_transform(X_data)
    joblib.dump(X_min_max_scaler, "../static/saved_models/X_min_max_scaler.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
    
    # print(weather_array[0])
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    joblib.dump(clf, "../static/saved_models/v2_rn_forest_"+str(pdq_number)+".pkl")
    print(clf.score(X_test, y_test))
    
    # mse
    y_pred = clf.predict(X_test)
    mse = sqrt(mean_squared_error(y_pred, y_test))
    print("rmse: "+str(mse))
    
    random_test_index = random.randrange(0, len(X_test))
    # print(random_test_index)
    
    # print(clf.predict([X_test[random_test_index]]))
    # print(y_test[random_test_index])
    
    predictions, actuals = [], []
    for i in range(len(y_pred)): 
        predictions.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])
        actuals.append(np.where(y_test[i] == np.max(y_test[i]))[0][0])
    
    categories = ["No_Crime", "Introduction", "Vol_dans_/_sur_véhicule_à_moteur", "Vol_de_véhicule_à_moteur", "Méfait", "Vols_qualifiés", "Infractions_entrainant_la_mort"]
    n_classes = len(categories)
    
    cm = sm.confusion_matrix(predictions, actuals, labels=[i for i in range(n_classes)])
    plot_confusion_matrix(cm, classes=categories, title='Confusion Matrix')
    plt.savefig("../static/saved_models/v2_rn_forest_"+str(pdq_number)+"_cnf_mtx.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    for i in range(len(categories)): 
        categories[i] = categories[i].replace("&", "and")
        categories[i] = categories[i].replace(" ", "_")
    
    cr = sm.classification_report(actuals, predictions, labels=[i for i in range(n_classes)], target_names=categories); cr = cr.split("\n")
    plot_classification_report(cr[0] + '\n\n' + cr[2] + '\n' + cr[3] + '\n' + cr[4] + '\n' + cr[5] + '\n' + cr[6] + '\n' + cr[7] + '\n' + cr[8] + '\n' + cr[10] + '\n', title = 'Classification Report')
    plt.savefig("../static/saved_models/v2_rn_forest_"+str(pdq_number)+"_cls_rep.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    pd.Series([ y[0] for y in y_pred ]).plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.xlabel('Values'); plt.ylabel('Frequency')
    plt.title('Predictions (y) Ranges Histogram')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("../static/saved_models/v2_rn_forest_"+str(pdq_number)+"_pred_dist.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
    
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
    plt.savefig("../static/saved_models/v2_rn_forest_"+str(pdq_number)+"_roc_auc.png", dpi=200, format='png', bbox_inches='tight'); plt.close();
>>>>>>> 5a0a78a4e2b9a024d210e41be3b341620b39547c
