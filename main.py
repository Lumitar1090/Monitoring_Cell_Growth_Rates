import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

### ------------------------------------
from src.classes import Cell_Growth_Rates
bm = Cell_Growth_Rates("datasets/dataset_cell_growth_4factor.csv")
bm.export("figures/Raw Data.png")
## -------------------------------------

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))
features = ["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]
targets = "Specific Growth Rate [d^-1]"

x = df.loc[:,features].to_numpy()
y = df.loc[:,targets].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Total samples:", len(x))
print("Training samples:", len(x_train))

compare_results = []
# -------------------------------------------------------------------------------------------------
### LINEAR REGRESSION ###

scaler = StandardScaler()
linreg = LinearRegression()

scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

linreg.fit(x_train_std, y_train)
y_train_pred = linreg.predict(x_train_std)
y_test_pred = linreg.predict(x_test_std)

m_train_rmse = root_mean_squared_error(y_train, y_train_pred)
m_train_r2 = r2_score(y_train, y_train_pred)
m_test_rmse = root_mean_squared_error(y_test, y_test_pred)
m_test_r2 = r2_score(y_test, y_test_pred)

print()
print("The Linear Regression gives:")
print()
print(f"Train | RMSE | {m_train_rmse:6.3} d^-1")
print(f"Train | R^2  | {m_train_r2:6.3}")
print(f"Test | RMSE | {m_test_rmse:6.3} d^-1")
print(f"Test | R^2  | {m_test_r2:6.3}")
print()

compare_results.append({"Model": "Linear Regression", "Train RMSE": m_train_rmse, "Train R^2":m_train_r2, "Test RMSE": m_test_rmse, "Test R^2":m_test_r2})

bm.plot_linear_regression(y_train, y_test, y_train_pred, y_test_pred,"figures/Linear_Regression_Actual_vs_Predicted.png", title="Linear Regression: Actual vs Predicted Growth Rate")
# _____________________________________________________________________________________________________

### POLYNOMIAL REGRESSION ###
#----- TESTING DEGREES -------#
poly_results = []
poly_predictions = {}
chosen_degree = 2

for degree in [2,3,4,5,6,7]:
    model = (make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()))
    model.fit(x_train, y_train)
    y_train_pred_poly = model.predict(x_train)
    y_test_pred_poly = model.predict(x_test)


    cv_results = cross_validate(model, x_train, y_train, cv=5, scoring=("neg_root_mean_squared_error", "r2"),
                                return_train_score=True)

    poly_train_rmse = -cv_results["train_neg_root_mean_squared_error"]
    poly_train_r2 = cv_results["train_r2"]
    poly_test_rmse = -cv_results["test_neg_root_mean_squared_error"]
    poly_test_r2 = cv_results["test_r2"]
    poly_predictions[degree] = (y_train_pred_poly, y_test_pred_poly)

    poly_results.append({"Degree": degree, "Train RMSE": np.mean(-cv_results["train_neg_root_mean_squared_error"]), "Train RMSE Std": np.std(-cv_results["train_neg_root_mean_squared_error"]), "Train R^2": np.mean(cv_results["train_r2"]), "Train R^2 Std": np.std(cv_results["train_r2"]), "Validation RMSE": np.mean(-cv_results["test_neg_root_mean_squared_error"]), "Validation RMSE Std": np.std(-cv_results["test_neg_root_mean_squared_error"]), "Validation R^2": np.mean(cv_results["test_r2"]), "Validation R^2 Std": np.std(cv_results["test_r2"])})

    print(f"Polynomial model with {degree = }:")
    print(f"  Train | RMSE | {np.mean(poly_train_rmse):6.1f} +/- {np.std(poly_train_rmse):6.1f} d^-1")
    print(f"  Validation  | RMSE | {np.mean(poly_test_rmse):6.1f} +/- {np.std(poly_test_rmse):6.1f} d^-1")
    print(f"  Train | R^2  | {np.mean(poly_train_r2):6.3f} +/- {np.std(poly_train_r2):6.3f}")
    print(f"  Validation  | R^2  | {np.mean(poly_test_r2):6.3f} +/- {np.std(poly_test_r2):6.3f}")
    print()

    if degree == chosen_degree: compare_results.append({"Model": "Polynomial Regression", "Train RMSE": root_mean_squared_error(y_train, y_train_pred_poly), "Train R^2": r2_score(y_train, y_train_pred_poly), "Test RMSE": root_mean_squared_error(y_test, y_test_pred_poly), "Test R^2":r2_score(y_test, y_test_pred_poly)})

poly_df = pd.DataFrame(poly_results)
poly_df.to_csv("tables/Polynomial_Degree_Comparison.csv", index=False)

bm.plot_polynomial_deg_comparison(y_train, y_test, poly_predictions, "figures/Polynomial_Degree_Comparison.png")


degree = 2

model = make_pipeline(StandardScaler(),PolynomialFeatures(degree=degree, include_bias=False),LinearRegression())
model.fit(x_train, y_train)

y_train_pred_poly = model.predict(x_train)
y_test_pred_poly = model.predict(x_test)

bm.plot_polynomial_regression(y_train, y_test,y_train_pred_poly, y_test_pred_poly,"figures/Polynomial_Degree_2_Actual_vs_Predicted.png",title="Polynomial Regression (Degree 2): Actual vs Predicted Growth Rate")
## --------------------------------------------------------------------------------------------------------

### KNN REGRESSION ###
model = make_pipeline(StandardScaler(),KNeighborsRegressor())
param_grid = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 81)], kneighborsregressor__weights=["uniform","distance"])

model_gscv= GridSearchCV(model, param_grid=param_grid, cv=4, scoring="neg_root_mean_squared_error", return_train_score=True)
model_gscv.fit(x_train, y_train)

y_train_pred_knn = model_gscv.predict(x_train)
y_test_pred_knn = model_gscv.predict(x_test)

df = pd.DataFrame(model_gscv.cv_results_)
df = df.sort_values(by=['rank_test_score'], ascending=True)

## KNN Validation ##
fig, ax = plt.subplots( figsize=(8,6), dpi=200, layout="constrained")

for weight in param_grid["kneighborsregressor__weights"]:
    mask_weight = df["param_kneighborsregressor__weights"] == weight
    df_sub = df.loc[mask_weight].sort_values(by="param_kneighborsregressor__n_neighbors")
    k = df_sub["param_kneighborsregressor__n_neighbors"].to_numpy()
    rmse_train = - df_sub["mean_train_score"].to_numpy()
    rmse_valid = - df_sub["mean_test_score"].to_numpy()

    ax.plot(k, rmse_train, label=f"Train (kNN: {weight})", color="pink", linestyle="-")
    ax.plot(k, rmse_valid, label=f"Validation (kNN: {weight})", color="orange", linestyle="--")

ax.set_xlabel("# of Neighbors")
ax.set_ylabel("RMSE [d^-1]")
ax.legend()
ax.set_ylim(bottom=0)

fig.savefig(os.path.join("figures", "knn_validation_curve.png"))
plt.close(fig)

columns_view = ["rank_test_score", "param_kneighborsregressor__n_neighbors", "param_kneighborsregressor__weights", "mean_test_score", "std_test_score", "mean_train_score", "std_train_score"]

print("The KNN Regression gives:")
print()
print(df[columns_view].head(10).to_string(index=False))
print()
print(f"Final Test RMSE: {-model_gscv.score(x_test, y_test):.6f} d^-1")
print()

compare_results.append({"Model": "KNN Regression", "Train RMSE": root_mean_squared_error(y_train, y_train_pred_knn), "Train R^2": r2_score(y_train, y_train_pred_knn), "Test RMSE": root_mean_squared_error(y_test, y_test_pred_knn), "Test R^2":r2_score(y_test, y_test_pred_knn)})
bm.plot_knn_regression(y_train, y_test, y_train_pred_knn, y_test_pred_knn,"figures/KNN_Regression_Actual_vs_Predicted.png", title="KNN Regression: Actual vs Predicted Growth Rate")

best_k = model_gscv.best_params_["kneighborsregressor__n_neighbors"]
best_weight = model_gscv.best_params_["kneighborsregressor__weights"]

best_knn = pd.DataFrame({"Parameter": ["Number of Neighbors (k)","Weighting Scheme","Cross-Validation RMSE","Test RMSE","Test R²"],
    "Value": [best_k, best_weight, f"{-model_gscv.best_score_:.4f}", f"{root_mean_squared_error(y_test, y_test_pred_knn):.4f}", f"{r2_score(y_test, y_test_pred_knn):.4f}"]})
best_knn.to_csv("tables/Best_KNN_Parameters.csv", index=False)

## COMPARISON SUMMARY ##
results_df = pd.DataFrame(compare_results)
print()
print("Comparison of Regression Models")
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

bm.save_metrics(results_df, "tables/Regression_Comparison.csv")