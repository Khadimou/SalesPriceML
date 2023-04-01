from sklearn.model_selection import cross_val_score, KFold, validation_curve, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Ridge, Lasso
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

house_dataset_df = pd.read_csv('../housing_dataset.csv')
Y =house_dataset_df['SalePrice']
house_dataset_df = house_dataset_df.drop(house_dataset_df.columns.difference(['OverallQual',
 'YearBuilt',
 'YearRemodAdd',
 'TotalBsmtSF',
 '1stFlrSF',
 'GrLivArea',
 'FullBath',
 'TotRmsAbvGrd',
 'GarageCars',
 'GarageArea',
 'MSSubClass', 'MSZoning','Neighborhood','Condition1']),axis=1)

house_dataset_df = house_dataset_df.values

#Transform data
labelEncoder = LabelEncoder()
Encoder = OrdinalEncoder()

for i in range(house_dataset_df.shape[1]):
    if type(house_dataset_df[0,i]) == str:
        res = Encoder.fit_transform(house_dataset_df[:,i].reshape(-1,1))
        house_dataset_df[:,i] = res[i]
        
X_processed = np.array(house_dataset_df)
#print(X_processed[9],Y[9])

cv = KFold(5)
# # Split the data into training and test sets
for train_index, test_index in cv.split(X_processed):
    xtrain, xtest = X_processed[train_index], X_processed[test_index]
    ytrain, ytest = Y[train_index], Y[test_index]

#Regression model
gbm = HistGradientBoostingRegressor(l2_regularization= 0.1,learning_rate= 0.2,max_depth= 2,min_samples_leaf= 2)
gbr = GradientBoostingRegressor()

# scores = cross_val_score(gbm, xtrain, ytrain)
# print("cross validation scores gbm= ",scores.mean())
# scores = cross_val_score(gbr, xtrain, ytrain)
# print("cross validation scores gbr= ",scores.mean())


model = gbr.fit(xtrain, ytrain)

# k = np.arange(0,100)
# train_score, val_score = validation_curve(gbr, xtrain, ytrain,param_name='learning_rate', param_range=k, cv=4)
# plt.plot(k,train_score.mean(axis=1), label='train')
# plt.plot(k,val_score.mean(axis=1), label='validation')
# plt.xlabel("learning_rate")
# plt.ylabel("score")
# plt.legend()
# plt.show()

# Define hyperparameter grid
# param_grid = {'max_depth': [2, 3, 4, 5],
#               'min_samples_split': [2, 3, 4]}

# # Create a grid search object
# grid_search = GridSearchCV(gbr, param_grid)

# # Fit the grid search object to the data
# grid_search.fit(xtrain, ytrain)

# # Print the best hyperparameters
# print(grid_search.best_params_)
# model = grid_search.best_estimator_
# print("model_score = ",model.score(xtest,ytest))
predictions = model.predict(xtest)
accuracy_lr = []

for i in range (0, ytest.shape[0]):
    if ytest.tolist()[i] - predictions[i] < 0:
        accuracy_lr.append(predictions[i] -ytest.tolist()[i])
    else:
        accuracy_lr.append(ytest.tolist()[i] - predictions[i])


accuracy_lr = np.asarray(accuracy_lr)
print("ecart(prediction,reel)= ",accuracy_lr.mean())


# N, train_score, val_score = learning_curve(model, xtrain, ytrain, train_sizes=np.linspace(0.2,1,10), cv=5)

# plt.plot(N,train_score.mean(axis=1), label='train')
# plt.plot(N,val_score.mean(axis=1), label='validation')
# plt.xlabel("train_sizes")
# plt.legend()
# plt.show()

# Set the parameters for the XGBoost model
# param = {
#     'max_depth': 3,
#     'eta': 0.3,
#     'objective': 'reg:squarederror'
# }
# num_round = 20
# # Convert the data into DMatrix format
# dtrain = xgb.DMatrix(xtrain, label=ytrain)
# dtest = xgb.DMatrix(xtest, label=ytest)
# # Train the XGBoost model
# bst = xgb.train(param, dtrain, num_round)

# # Make predictions on the test set
# ypred = bst.predict(dtest)
# # Calculate the RMSE
# rmse = np.sqrt(mean_squared_error(ytest, ypred))
# print(f'RMSE: {rmse:.2f}')

# # Calculate the R-squared score
# r2 = r2_score(ytest, ypred)
# print(f'R-squared: {r2:.2f}')

# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# # Calculate the MAPE
# mape = mean_absolute_percentage_error(ytest, ypred)
# print(f'MAPE: {mape:.2f}%')

# # Perform cross-validation and record the performance metrics
# cv_results = xgb.cv(
#     param,
#     dtrain,
#     num_round,
#     nfold=5,
#     metrics='rmse',
#     early_stopping_rounds=10
# )

# # Plot the learning curve
# plt.plot(cv_results['train-rmse-mean'], label='Train')
# plt.plot(cv_results['test-rmse-mean'], label='Test')
# plt.xlabel('Round')
# plt.ylabel('rmse')
# plt.legend()
#plt.show()


import pickle
#Save model
pickle.dump(model, open('model.pkl', 'wb'))


