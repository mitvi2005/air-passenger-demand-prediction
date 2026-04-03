import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import TimeSeriesSplit
df=pd.read_csv("air_passengers_variable_season_2026 (1).csv")
df.head(15)
df.info()
df.isnull().sum()
# Here Month Values is object so convert it into date and time
df['Month']=pd.to_datetime(df['Month'],errors='coerce',dayfirst=True)
df.info()
df.describe()
df['Passengers']
# Here Passengers is also in object so it not valid convert it into numeric form
df['Passengers']=pd.to_numeric(df['Passengers'],errors='coerce')
df.isnull().sum()
df.loc[df['Passengers'] < 0, 'Passengers'] = np.nan
df.isnull().sum()
df['Passengers'].describe()
df.isnull().sum()
df.isnull().sum()
sns.boxplot(df['Passengers'])
plt.show()
plt.scatter(df['Month'],df['Passengers'])
plt.show()
Q1 = df['Passengers'].quantile(0.25)
Q3 = df['Passengers'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Passengers'] < lower_bound) | (df['Passengers'] > upper_bound)]
print("Outliers:\n", outliers)
df = df[(df['Passengers'] >= lower_bound) & (df['Passengers'] <= upper_bound)]
print("Lower:", lower_bound)
print("Upper:", upper_bound)
# df['Passengers']=df['Passengers'].fillna(method='ffill')
# df['Passengers'] = df['Passengers'].interpolate(method='linear')
df = df.dropna(axis=1, how='all')
(df['Passengers']<0).sum()
sns.boxplot(df['Passengers'])
plt.show()
df.describe()
df.info()
plt.scatter(df['Month'], df['Passengers'])
plt.show()
df['month']=df['Month'].dt.month
df['year']=df['Month'].dt.year
df.info()
df.drop('Month',axis=1,inplace=True)
df.head()
df.isnull().sum()
plt.scatter(df['year'],df['Passengers'])
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
plt.scatter(df['month'],df['Passengers'])
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['year'],df['Passengers'])
plt.title("Air Passengers Trend")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()
plt.figure(figsize=(10,5))
plt.plot(df['year'],df['Passengers'])
plt.title("Air Passengers Trend")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()
sns.heatmap(df.corr(),annot=True)
plt.show()
x=df.drop('Passengers',axis=1)
y=df['Passengers']
def evaluate(y_test,pred):

    print("MAE:",mean_absolute_error(y_test,pred))
    print("RMSE:",np.sqrt(mean_squared_error(y_test,pred)))
    print("R2 Score:",r2_score(y_test,pred))
def print_results(name, model):
    print("Model:", name)
    print("Best Estimator:", model.best_estimator_)
    print("Best Parameters:", model.best_params_)
    print("Best CV Score:", model.best_score_)
tscv = TimeSeriesSplit(n_splits=5)
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)
scores=[]
labels=[]
scores1=[]
labels1=[]
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(x_train)
X_test_sc = scaler.transform(x_test)
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
pred_lr = lr.predict(x_test)
scores.append(r2_score(y_test,pred_lr))
labels.append("LR")
scores1.append(r2_score(y_test,pred_lr))
labels1.append("LR")
evaluate(y_test,pred_lr)
future = pd.DataFrame({
'month':[6],
'year':[2025]
})

# future_scaled = scaler.transform(future)
# future_poly = poly.transform(future_scaled)

pred_future =lr.predict(future)

print(pred_future)
degress=[1,2,3,4,5]
best_score=-1
score=[]
degree=[]
colors=['blue','red','green','pink','orange']
for d in degress:
    poly=PolynomialFeatures(degree=d)
    x_train_poly=poly.fit_transform(x_train)
    x_test_poly=poly.transform(x_test)
    model=LinearRegression()
    model.fit(x_train_poly,y_train)
    y_pred=model.predict(x_test_poly)
    score.append(r2_score(y_test,y_pred))
    degree.append(d)
plt.bar(degree,score,color=colors)
plt.show()
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_sc)
X_test_poly = poly.transform(X_test_sc)
model_lr = LinearRegression()
model_lr.fit(X_train_poly,y_train)
pred_poly = model_lr.predict(X_test_poly)
scores.append(r2_score(y_test,pred_poly))
labels.append("Poly")
scores1.append(r2_score(y_test,pred_poly))
labels1.append("Poly")
evaluate(y_test,pred_lr)
future = pd.DataFrame({
'month':[5],
'year':[1949]
})

future_scaled = scaler.transform(future)
future_poly = poly.transform(future_scaled)

pred_future =model_lr.predict(future_poly)

print(pred_future)
plt.figure(figsize=(6,6))

plt.scatter(y_test, pred_poly)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Polynomial Regression (Degree 3) - Actual vs Predicted")

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.show()
plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Actual")
plt.plot(pred_poly, label="Predicted")

plt.title("Polynomial Regression (Degree 3)")
plt.xlabel("Index")
plt.ylabel("Passengers")
plt.legend()

plt.show()
ridge = Ridge()
ridge.fit(X_train_poly,y_train)

pred_ridge = ridge.predict(X_test_poly)
evaluate(y_test,pred_ridge)
scores.append(r2_score(y_test,pred_ridge))
labels.append("Ridge")
param_grid = {'alpha':[0.01,0.1,1,10,100]}

grid_ridge = GridSearchCV( ridge, param_grid, cv=tscv, scoring='r2')

grid_ridge.fit(X_train_poly,y_train)

print_results("Ridge",grid_ridge)

pred_ridge_grid = grid_ridge.predict(X_test_poly)

evaluate(y_test,pred_ridge_grid)
scores1.append(r2_score(y_test,pred_ridge_grid))
labels1.append("Ridge-Grid")
scores1

lasso = Lasso()
lasso.fit(X_train_poly,y_train)

pred_lasso =lasso.predict(X_test_poly)
scores.append(r2_score(y_test,pred_lasso))
labels.append("Lasso")
evaluate(y_test,pred_lasso)
param_grid = {'alpha':[0.001,0.01,0.1,1,10]}

grid_lasso = GridSearchCV( lasso, param_grid, cv=tscv, scoring='r2'
)

grid_lasso.fit(X_train_poly,y_train)

print_results("Lasso",grid_lasso)

pred_lasso_grid = grid_lasso.predict(X_test_poly)

evaluate(y_test,pred_lasso_grid)
scores1.append(r2_score(y_test,pred_lasso_grid))
labels1.append("Lasso-Grid")
elastic = ElasticNet()
elastic.fit(X_train_poly,y_train)
pred_elastic =elastic.predict(X_test_poly)
evaluate(y_test,pred_elastic)
scores.append(r2_score(y_test,pred_elastic))
labels.append("Elastic")
param_grid = {
    'alpha':[0.01,0.1,1,10,0.001],
    'l1_ratio':[0.2,0.5,0.7]
}

grid_elastic = GridSearchCV( elastic, param_grid, cv=tscv, scoring='r2'
)

grid_elastic.fit(X_train_poly,y_train)

print_results("ElasticNet",grid_elastic)

pred_elastic_grid = grid_elastic.predict(X_test_poly)

evaluate(y_test,pred_elastic_grid)
scores1.append(r2_score(y_test,pred_elastic_grid))
labels1.append("Elastic-Grid")
tree = DecisionTreeRegressor()
tree.fit(x_train,y_train)
pred_tree =tree.predict(x_test)
evaluate(y_test,pred_tree)
scores.append(r2_score(y_test,pred_tree))
labels.append("Decision-Tree")
param_grid = {
    'max_depth':[3,5,10],
    'min_samples_split':[2,5,10]
}

grid_tree = GridSearchCV( tree, param_grid, cv=tscv, scoring='r2')

grid_tree.fit(x_train,y_train)

print_results("Decision Tree",grid_tree)

pred_tree_grid = grid_tree.predict(x_test)

evaluate(y_test,pred_tree_grid)
scores1.append(r2_score(y_test,pred_tree_grid))
labels1.append("Decision-Tree")
rf = RandomForestRegressor()
rf.fit(x_train,y_train)
pred_rf =rf.predict(x_test)
evaluate(y_test,pred_rf)
scores.append(r2_score(y_test,pred_rf))
labels.append("Random-Forest")
param_dist = {
    'n_estimators':[100,200,300,400],
    'max_depth':[None,5,10,15],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

random_rf = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=20,
    cv=tscv,
    scoring='r2',
    random_state=42
)

random_rf.fit(x_train,y_train)

print_results("Random Forest",random_rf)

pred_rf_rand = random_rf.predict(x_test)

evaluate(y_test,pred_rf_rand)
scores1.append(r2_score(y_test,pred_rf_rand))
labels1.append("Random-Forest")
svr = SVR(kernel='linear')
svr.fit(X_train_sc,y_train)

pred_svr = svr.predict(X_test_sc)
evaluate(y_test,pred_svr)
scores.append(r2_score(y_test,pred_svr))
labels.append("SVR")

param_dist = {
    'C':[0.1,1,10,100],
    'gamma':['scale','auto'],
    'kernel':['rbf','linear']
}

random_svr = RandomizedSearchCV(
    svr,
    param_dist,
    n_iter=10,
    cv=tscv,
    scoring='r2'
)
random_svr.fit(X_train_sc,y_train)
print_results("SVR",random_svr)
pred_svr_rand = random_svr.predict(X_test_sc)
evaluate(y_test,pred_svr_rand)
scores1.append(r2_score(y_test,pred_svr_rand))
labels1.append("SVR")
knn = KNeighborsRegressor()
knn.fit(X_train_sc,y_train)

pred_knn =knn.predict(X_test_sc)
evaluate(y_test,pred_knn)
scores.append(r2_score(y_test,pred_knn))
labels.append("KNN")
future = pd.DataFrame({
'month':[10],
'year':[2020]
})
future_scaled = scaler.transform(future)
pred_future =knn.predict(future_scaled)
print(pred_future)
param_grid = {
    'n_neighbors':[3,5,7,9,11]
}

grid_knn = GridSearchCV(
    knn,
    param_grid,
    cv=tscv,
    scoring='r2'
)

grid_knn.fit(x_train,y_train)

print_results("KNN",grid_knn)

pred_knn_grid = grid_knn.predict(x_test)

evaluate(y_test,pred_knn_grid)
scores1.append(r2_score(y_test,pred_knn_grid))
labels1.append("KNN")
future = pd.DataFrame({
'month':[6],
'year':[2025]
})

# future_scaled = scaler.transform(future)
# future_poly = poly.transform(future_scaled)

pred_future =grid_knn.predict(future)

print(pred_future)
# ================= BOOSTING METHODS ================= #

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

# ---------- ADA BOOST ---------- #
ada = AdaBoostRegressor(random_state=42)
ada.fit(x_train, y_train)

pred_ada = ada.predict(x_test)

print("\n--- AdaBoost ---")
evaluate(y_test, pred_ada)

scores.append(r2_score(y_test, pred_ada))
labels.append("AdaBoost")

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}

grid_ada = GridSearchCV(
    ada,
    param_grid,
    cv=tscv,
    scoring='r2'
)

grid_ada.fit(x_train, y_train)

print_results("AdaBoost", grid_ada)

pred_ada_grid = grid_ada.predict(x_test)

evaluate(y_test, pred_ada_grid)

scores1.append(r2_score(y_test, pred_ada_grid))
labels1.append("AdaBoost")


# ---------- GRADIENT BOOSTING ---------- #
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(x_train, y_train)

pred_gbr = gbr.predict(x_test)

print("\n--- Gradient Boosting ---")
evaluate(y_test, pred_gbr)

scores.append(r2_score(y_test, pred_gbr))
labels.append("GradientBoost")

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

grid_gbr = GridSearchCV(
    gbr,
    param_grid,
    cv=tscv,
    scoring='r2'
)

grid_gbr.fit(x_train, y_train)

print_results("GradientBoost", grid_gbr)

pred_gbr_grid = grid_gbr.predict(x_test)

evaluate(y_test, pred_gbr_grid)

scores1.append(r2_score(y_test, pred_gbr_grid))
labels1.append("GradientBoost")


# ---------- XGBOOST ---------- #
from xgboost import XGBRegressor

xgb = XGBRegressor(random_state=42)
xgb.fit(x_train, y_train)

pred_xgb = xgb.predict(x_test)

print("\n--- XGBoost ---")
evaluate(y_test, pred_xgb)

scores.append(r2_score(y_test, pred_xgb))
labels.append("XGBoost")

# Hyperparameter Tuning
param_dist = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7]
}

random_xgb = RandomizedSearchCV(
    xgb,
    param_dist,
    n_iter=10,
    cv=tscv,
    scoring='r2',
    random_state=42
)

random_xgb.fit(x_train, y_train)

print_results("XGBoost", random_xgb)

pred_xgb_rand = random_xgb.predict(x_test)

evaluate(y_test, pred_xgb_rand)

scores1.append(r2_score(y_test, pred_xgb_rand))
labels1.append("XGBoost")
# ================= ADA BOOST WITH LINEAR REGRESSION ================= #

# Create AdaBoost with Linear Regression as base model
ada_lr = AdaBoostRegressor(
    estimator=LinearRegression(),   # 👈 Using LR instead of tree
    n_estimators=50,
    learning_rate=0.1,
    random_state=42
)

# Train model
ada_lr.fit(x_train, y_train)

# Predict
pred_ada_lr = ada_lr.predict(x_test)

print("\n--- AdaBoost with Linear Regression ---")

# Evaluate
evaluate(y_test, pred_ada_lr)

# Store results (same as your structure)
scores.append(r2_score(y_test, pred_ada_lr))
labels.append("AdaBoost-LR")
param_grid = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 1]
}

grid_ada_lr = GridSearchCV(
    ada_lr,
    param_grid,
    cv=tscv,
    scoring='r2'
)

grid_ada_lr.fit(x_train, y_train)

print_results("AdaBoost-LR", grid_ada_lr)

pred_ada_lr_grid = grid_ada_lr.predict(x_test)

evaluate(y_test, pred_ada_lr_grid)

scores1.append(r2_score(y_test, pred_ada_lr_grid))
labels1.append("AdaBoost-LR")
# ================= COMPARISON: ADA TREE vs ADA LR ================= #

from sklearn.tree import DecisionTreeRegressor

# AdaBoost with Decision Tree (default / weak learner)
ada_tree = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),  # weak learner (stump)
    n_estimators=50,
    learning_rate=0.1,
    random_state=42
)

ada_tree.fit(x_train, y_train)
pred_ada_tree = ada_tree.predict(x_test)

print("\n--- AdaBoost with Decision Tree ---")
evaluate(y_test, pred_ada_tree)

# Store
tree_score = r2_score(y_test, pred_ada_tree)
lr_score = r2_score(y_test, pred_ada_lr)


# ================= GRAPH COMPARISON ================= #

models = ["AdaBoost-Tree", "AdaBoost-LR"]
scores_compare = [tree_score, lr_score]

plt.figure(figsize=(6,5))
plt.bar(models, scores_compare)
plt.title("AdaBoost: Tree vs Linear Regression")
plt.xlabel("Model")
plt.ylabel("R2 Score")

# show values
for i, v in enumerate(scores_compare):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.show()
base_models = {
    "Decision Tree (Stump)": DecisionTreeRegressor(max_depth=1),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=50),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

print("AdaBoost with Different Base Models (R2 Scores):\n")
for name, base_model in base_models.items():
    
    ada = AdaBoostRegressor(
        estimator=base_model,
        n_estimators=50,
        learning_rate=0.1,
        random_state=42
    )
    
    try:
        if name in ["SVR", "KNN"]:
            ada.fit(X_train_sc, y_train)
            pred = ada.predict(X_test_sc)
        else:
            ada.fit(x_train, y_train)
            pred = ada.predict(x_test)
        
        score = r2_score(y_test, pred)
        print(f"{name}: {score:.5f}")
    
    except Exception as e:
        print(f"{name}:Error -> {str(e)}")
# colors = ['red','blue','green','orange','purple','cyan','brown','pink']
plt.figure(figsize=(14,6))
plt.bar(labels,scores,color=colors)
plt.title("Model Accuracy Without Hyperparameter Tuning")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.savefig('WithoutTuning.jpg')
plt.show()
# colors = ['red','blue','green','orange','purple','cyan','brown','pink']
plt.figure(figsize=(14,6))
plt.bar(labels,scores,color=colors)
plt.title("Model Accuracy Without Hyperparameter Tuning")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.savefig('WithoutTuning.jpg')
plt.show()
scores1
# colors = ['red','blue','green','orange','purple','cyan','brown','pink']
plt.figure(figsize=(14,6))
plt.bar(labels1,scores1)
plt.title("Model Accuracy With Hyperparameter Tuning")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.savefig('WithTuning.jpg')
plt.show()
# ================= CSS-LIKE STYLING ================= #

plt.style.use('seaborn-v0_8-darkgrid')   # theme

colors = ['#ff6b6b','#4dabf7','#51cf66','#ffa94d','#845ef7','#22b8cf','#7950f2','#f06595']

plt.figure(figsize=(14,6), facecolor='#f8f9fa')

bars = plt.bar(labels1, scores1, color=colors, edgecolor='black', linewidth=1.2)

# Title styling
plt.title("Model Accuracy With Hyperparameter Tuning",
          fontsize=18, fontweight='bold', color='#343a40')

# Axis labels
plt.xlabel("Models", fontsize=12, fontweight='bold')
plt.ylabel("R2 Score", fontsize=12, fontweight='bold')

# Rotate labels for better UI
plt.xticks(rotation=30, fontsize=10)
plt.yticks(fontsize=10)

# Add values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f'{height:.2f}',
             ha='center',
             va='bottom',
             fontsize=10,
             fontweight='bold')

# Remove top/right borders (clean UI look)
for spine in ['top','right']:
    plt.gca().spines[spine].set_visible(False)

# Grid styling
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('WithTuning_styled.jpg', dpi=300)
plt.show()
plt.scatter(y_test,pred_knn)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
plt.figure(figsize=(12,7))
plt.plot(y_test.values,label="Actual")
plt.plot(pred_knn,label="Predicted")

plt.legend()
plt.show()
import pickle
pickle.dump(lr, open("model.pkl", "wb"))