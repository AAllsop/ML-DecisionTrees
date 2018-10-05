import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

pd.options.mode.chained_assignment = None  
working_path = r"C:\Users\allsopa\OneDrive - City Holdings\Personal\ML Practice\Python\Decision Trees\Gini Example" + "\\"

#import files
faults = pd.read_csv(working_path + "faults.csv")
lookup_keys_callertype = pd.read_csv(working_path + "lookup_keys_callertype.csv")
lookup_keys_priority = pd.read_csv(working_path + "lookup_keys_priority.csv")
lookup_keys_service = pd.read_csv(working_path + "lookup_keys_service.csv")

faults.index = faults["FaultID"]

faultsdf = pd.merge(faults,lookup_keys_callertype, left_on = "CallerTypeKey",right_on = "CallerTypeKey", how = "inner")
faultsdf = pd.merge(faultsdf,lookup_keys_service, left_on = "ServiceKey", right_on = "ServiceKey", how = "inner")
faultsdf = pd.merge(faultsdf,lookup_keys_priority, left_on = "PriorityKey", right_on = "PriorityKey", how = "inner" )

faults_x = faultsdf[["CallerType","CalloutOutOfHours","Service"]]
faults_y = faultsdf[["Priority"]]

#numerical encoding
le = preprocessing.LabelEncoder()
faults_x_num = faults_x.apply(le.fit_transform)
faults_y_num = faults_y.apply(le.fit_transform)

#split between training and testing
x_train, x_test, y_train, y_test = model_selection.train_test_split(faults_x_num,faults_y,test_size = 0.2, random_state = 1)

#initialise model
#   gini
model_gini = tree.DecisionTreeClassifier(criterion="gini", random_state=1)
model_gini = model_gini.fit(x_train,y_train)

#prepare dot data for graphing.
#need to export this and open manually in dot data graphing tool as Python can't locate the .exe for GrpahViz
col_names = faults_x.columns
dot_data_gini = tree.export_graphviz(model_gini, out_file=None,
                                     feature_names = col_names,
                                     class_names = np.unique(faults_y.Priority),
                                     filled=True, rounded=True)

#Make prediction on test data
#   gini
y_pred_gini = model_gini.predict(x_test)
y_test["y_pred_gini"] = y_pred_gini

#calculate accuracy (% of right predctions to all records)
print("Gini accuracy: ", metrics.accuracy_score(y_test["Priority"],y_pred_gini)*100)

#construct confusion matrix
z = pd.crosstab(y_test["Priority"],y_pred_gini, rownames = ["Actual Values"], colnames = ["Predicted Values"])









