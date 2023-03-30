from sklearn.model_selection import cross_val_score, KFold, validation_curve, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

house_dataset = pd.read_csv('../housing_dataset.csv')
house_dataset_df = pd.DataFrame(house_dataset)



dtr = DecisionTreeRegressor()
lr = LinearRegression()
ridge = Ridge(alpha=0.1)
tree = DecisionTreeRegressor()
# Create a random forest regressor
forest = RandomForestRegressor(max_depth=2, random_state=0)


# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['OverallQual',
 'YearBuilt',
 'YearRemodAdd',
 'TotalBsmtSF',
 '1stFlrSF',
 'GrLivArea',
 'FullBath',
 'BsmtFullBath',
 'TotRmsAbvGrd',
 'GarageCars',
 'GarageArea',
 'SalePrice','Fireplaces',]),
        ('cat', OneHotEncoder(), ['MSSubClass','MSZoning','Neighborhood','Condition1','RoofStyle',	
'BsmtQual',	
'BsmtExposure',	
'HeatingQC','CentralAir','KitchenQual','FireplaceQu',	
'GarageType',	
'GarageFinish',	'PavedDrive',	
'SaleCondition'])
    ])

# Transform the data
X_processed = preprocessor.fit_transform(house_dataset_df)
cv = KFold(5, random_state=0, shuffle=True)
xtrain, xtest, ytrain, ytest = train_test_split(X_processed, house_dataset_df['SalePrice'], train_size=0.8)
cross_val_score(ridge, xtrain, ytrain, cv=cv).mean()

ridge.fit(xtrain, ytrain)
predictions = ridge.predict(xtest)
erreur = 1 - ridge.score(xtest,ytest)
accuracy_lr = []

for i in range (0, ytest.shape[0]):
    if ytest.tolist()[i] - predictions[i] < 0:
        accuracy_lr.append(predictions[i] -ytest.tolist()[i])
    else:
        accuracy_lr.append(ytest.tolist()[i] - predictions[i])


accuracy_lr = np.asarray(accuracy_lr)
print(accuracy_lr.mean())

k = np.arange(0,10)
train_score, val_score = validation_curve(ridge, xtrain, ytrain,param_name='alpha', param_range=k, cv=4)
plt.plot(k,train_score.mean(axis=1), label='train')
plt.plot(k,val_score.mean(axis=1), label='validation')
plt.xlabel("alpha")
plt.ylabel("score")
plt.legend()
plt.show()
# Define hyperparameter grid
param_grid = {'alpha': [0.1, 0.01, 0.0009, 0.2, 0.05]}

# Create a grid search object
grid_search = GridSearchCV(ridge, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(xtrain, ytrain)

# Print the best hyperparameters
print(grid_search.best_params_)
model = grid_search.best_estimator_
print("model_score = ",model.score(xtest,ytest))

N, train_score, val_score = learning_curve(model, xtrain, ytrain, train_sizes=np.linspace(0.2,1,10), cv=5)
print(N)
plt.plot(N,train_score.mean(axis=1), label='train')
plt.plot(N,val_score.mean(axis=1), label='validation')
plt.xlabel("train_sizes")
plt.legend()
plt.show()

# Convert the data into DMatrix format
dtrain = xgb.DMatrix(xtrain, label=ytrain)
dtest = xgb.DMatrix(xtest, label=ytest)

# Set the parameters for the XGBoost model
param = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'reg:squarederror'
}
num_round = 20

# Train the XGBoost model
bst = xgb.train(param, dtrain, num_round)

# Make predictions on the test set
ypred = bst.predict(dtest)
# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(ytest, ypred))
print(f'RMSE: {rmse:.2f}')

# Calculate the R-squared score
r2 = r2_score(ytest, ypred)
print(f'R-squared: {r2:.2f}')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate the MAPE
mape = mean_absolute_percentage_error(ytest, ypred)
print(f'MAPE: {mape:.2f}%')

# Perform cross-validation and record the performance metrics
cv_results = xgb.cv(
    param,
    dtrain,
    num_round,
    nfold=5,
    metrics='rmse',
    early_stopping_rounds=10
)

# Plot the learning curve
plt.plot(cv_results['train-rmse-mean'], label='Train')
plt.plot(cv_results['test-rmse-mean'], label='Test')
plt.xlabel('Round')
plt.ylabel('rmse')
plt.legend()
plt.show()