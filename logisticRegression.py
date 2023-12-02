
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# roc curve and auc
from sklearn.datasets import make_classification

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


train_df = pd.read_csv(r'C:\Users\Ropa\Desktop\project\yield3.csv')
train_df.dropna(inplace=True)
print(train_df.head())

crop_data = pd.get_dummies(train_df['Crop_Name'], drop_first = True)
print(crop_data)
train_df = pd.concat([train_df, crop_data], axis = 1)
print(train_df)

#train_df.drop(['Crop_Name'], axis = 1, inplace = True)
#print(train_df)

features= train_df[['Rainfall', 'Humidity', 'Temperature', 'Pesticides', 'Soil_ph',
'N','P','K','Area_Planted']]
tests = train_df['Yams']
X_train , X_test , y_train , y_test = train_test_split(features , tests ,test_size = 0.3)

scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(train_features , y_train)
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_predict = model.predict(test_features)

print('\n')

print("The training score of the model is: ", train_score)
print('\n')
print("The score of the model on test data is:", test_score )
print('\n')
print("The predicted output array is:",y_predict)

print('\n')
# save the model to disk
import pickle
filename = 'logistic_reg.pkl'
pickle.dump(filename, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


confusion = confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]


# predict probabilities on Test and take probability for class 1([:1])
y_pred_prob_test = model.predict_proba(X_test)[:, 1]
#predict labels on test dataset
y_pred_test = model.predict(X_test)
# create onfusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix is ",confusion_matrix)

#Visuallizing confusion matrix in heatmap
plt.title("Confusion Matrix Heat Map")
plt.ylabel('Real Values')
plt.xlabel('Predicted Values')
sns.heatmap(confusion_matrix, annot = True)
plt.show()


# importing accuracy score
from sklearn.metrics import accuracy_score
# printing the accuracy of the model
print("Accuracy score of the model is ", accuracy_score(y_test, y_predict))
print('\n')


# ROC- AUC score
#print("ROC-AUC score ", roc_auc_score(y_test,y_pred_prob_test))
#Precision score
#print("precision score ", precision_score(y_test,y_pred_test))
#Recall Score
#print("Recall score ", recall_score(y_test,y_pred_test))
#f1 score
#print("f1 score ", f1_score(y_test,y_pred_test))



from sklearn.metrics import classification_report
print("Classification Report is \n", classification_report(y_test, y_predict))



trainX, testX, trainy, testy = train_test_split(features , tests, test_size=0.5, random_state=2)
scaler = StandardScaler()
features = scaler.fit_transform(trainX)
tests = scaler.transform(testX)

# making classifications
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

# generating a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fitting a model
model = LogisticRegression()
model.fit(trainX, trainy)
# predicting probabilities
lr_probs = model.predict_proba(testX)
# keeping probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculating no skill and logistic regression roc_auc scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarizing scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculating roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plotting the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# showing the legend
plt.legend()
# showing the plot
plt.show()




# predicting class values (precision and recall)
yhat = model.predict(testX)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarizing scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plotting the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# showing the legend
plt.legend()
# showing the plot
plt.show()




