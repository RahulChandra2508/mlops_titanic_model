import pipeline as pipe
import data_managers
from sklearn.metrics import roc_auc_score, accuracy_score

#loading the training data
X_train, y_train = data_managers.load_data(split= True, data_= "train")

titanic_pipe = pipe.pipeline()

titanic_pipe.fit(X_train, y_train)

# make predictions for train set
class_ = titanic_pipe.predict(X_train)
pred = titanic_pipe.predict_proba(X_train)[:,1]
# determine roc and accuracy
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))

#saving the model
data_managers.save_model(titanic_pipe)

