import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tbl import feature

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree,DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

import warnings
warnings.filterwarnings('ignore')

#1
df=pd.read_csv('diabetes.csv')
df_name=df.columns

df.info()

describe=df.describe()

sns.pairplot(df,hue="Outcome")
plt.show()

def plot_correlation_heatmap(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

plot_correlation_heatmap(df)


def detect_outliers_iqr(df):
    outlier_indices= []
    outliers_df=pd.DataFrame()
    for col in df.select_dtypes(include=["float64","int64"]).columns:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)

        IQR=Q3-Q1
        lower_bound=Q1-1.5*IQR
        upper_bound=Q3+1.5*IQR

        outliers_in_col=df[df[col]<lower_bound | df[col]>upper_bound]
        outlier_indices.extend(outliers_in_col.index)
        outliers_df = pd.concat([outliers_df,outliers_in_col],axis=0)


    outlier_indices=list(set(outlier_indices))
    outliers_df=outliers_df.drop_duplicates()


    return outlier_indices, outliers_df

outlier_indices,outliers_df = detect_outliers_iqr(df)
df_cleaned=df.drop(outlier_indices).reset_index(drop=True)


X=df_cleaned.drop("Outcome",axis=1)
y=df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


def getBaseModel():
    based_models=[]
    based_models.append(('LR', LogisticRegression()))
    based_models.append(('DT', DecisionTreeClassifier()))
    based_models.append(('NB', GaussianNB()))
    based_models.append(('KNN', KNeighborsClassifier()))
    based_models.append(('AdaBoost', AdaBoostClassifier()))
    based_models.append(('GBDT', GradientBoostingClassifier()))
    based_models.append(('RF', RandomForestClassifier()))
    return based_models


def baseModelTraining(X_train, X_test, models):

    results=[]
    names=[]
    for name, model in models:
        kfold=KFold(n_splits=10, random_state=42, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name} Accuracy: {cv_results.mean()},std: {cv_results.std()}")

    return names,results

def plot_box(names,results):
    df=pd.DataFrame({names[i]:results[i] for i in range(len(names))})
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df)
    plt.title("Model accuracy")
    plt.show()

models=getBaseModel()
names,results=baseModelTraining(X_train, X_test, models)
plot_box(names,results)

param_grid={
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt=DecisionTreeClassifier()
grid_search=GridSearchCV(estimator=dt, param_grid=param_grid,cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("best params:", grid_search.best_params_)
best_dt_model=grid_search.best_estimator_

y_pred = best_dt_model.predict(X_test)

print("confusion matrix")
print(confusion_matrix(y_test, y_pred))

print("classification report")
print(classification_report(y_test, y_pred))

new_data=np.array([[6, 149, 72, 35, 0, 34.6, 0.627, 51]])
new_prediction=best_dt_model.predict(new_data)
print("new prediction", new_prediction)









#2
df=pd.read_csv('heart_disease_uci.csv')
df=df.drop(columns=['id'])

df.info()
describe=df.describe()

numerical_features=df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure()
sns.pairplot(df, vars=numerical_features, hue="num")
plt.show()

plt.figure()
sns.countplot(x="num",data=df)
plt.show()

print(df.isnull().sum())
df=df.drop(columns=["ca"])
print(df.isnull().sum())

df["trestbps"].fillna(df["trestbps"].mean(), inplace=True)
df["chol"].fillna(df["chol"].mean(), inplace=True)
df["fbs"].fillna(df["fbs"].mean(), inplace=True)
df["restecg"].fillna(df["restecg"].mean(), inplace=True)
df["thalch"].fillna(df["thalch"].mean(), inplace=True)
df["exang"].fillna(df["exang"].mean(), inplace=True)
df["oldpeak"].fillna(df["oldpeak"].mean(), inplace=True)
df["slope"].fillna(df["slope"].mean(), inplace=True)
df["thal"].fillna(df["thal"].mean(), inplace=True)

print(df.isnull().sum())

X=df.drop("num", axis=1)
y=df["num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features=["sex", "dataset", "cp", "restecg", "exang", "slop", "thal"]
numerical_features=["age", "trestbps", "chol", "fbs", "thalch", "oldpeak"]

X_train_num=X_train[numerical_features]
X_test_num=X_test[numerical_features]

scaler=StandardScaler()
X_train_num_scaled=scaler.fit_transform(X_train_num)
X_test_num_scaled=scaler.transform(X_test_num)

encoder=OneHotEncoder(sparse_output=False, drop="first")
X_train_cat=X_train[categorical_features]
X_test_cat=X_test[categorical_features]

X_train_cat_encoded=encoder.fit_transform(X_train_cat)
X_test_cat_encoded=encoder.transform(X_test_cat)

x_train_transformed=np.hstack((X_train_cat_encoded, X_train_num_scaled))
x_test_transformed=np.hstack((X_test_cat_encoded, X_test_num_scaled))


rf=RandomForestClassifier(n_estimators=100, random_state=42)
knn=KNeighborsClassifier()

voting_clf=VotingClassifier(estimators=[('rf', rf), ('knn', knn)],voting='soft')

voting_clf.fit(x_train_transformed, y_train)
y_pred = voting_clf.predict(x_test_transformed)

print("Accuracy", accuracy_score(y_test, y_pred))
print("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification Report")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()









#3
df = pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv")
df.info()

decribe=df.describe()
loss = df["Lenght of Stay"]
df["Lenght of Stay"] = df["Lenght of Stay"].replace("120 +", 120)
loss = df["Lenght of Stay"]

df.isna().sum()

for column in df.columns:
    unique_values = len(df[column].unique())
    print(f"Number of unique values in {column}: {unique_values}")

df = df[df["Patient Disposition"] != "Expired"]

f, ax = plt.subplots()
sns.boxplot(x="Payment Type", y="Lenght of Stay", data=df)
plt.title("Payment Type vs Lenght of Stay")
plt.xticks(rotation=60)

f, ax = plt.subplots()
sns.boxplot(x="Age Group", data = df[df["Payment Typology 1"] == "Medicare"], order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or older"])
plt.title("Medicare Partients for Age group")

f, ax = plt.subplots()
sns.boxplot(x="type of Admission", y="Lenght of Stay", data=df)
plt.title("Type of Admission vs Lenght of Stay")
plt.xticks(rotation=60)

f, ax = plt.subplots()
sns.boxplot(x="Age Group", y="Lenght of Stay", data=df, order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or older"])
plt.title("Age Group vs Lenght of Stay")
plt.xticks(rotation=60)
ax.set(ylim=(0, 25))


df = df.drop(["Hospital Service Area", "Hospital Country", "Operating Certificate Number",
              "Facility Name", "Zip Code - 3 digits", "Patient Disposition","Discharge Year"
              "CCSR Diagnosis Description", "CCSR Procedure Description", "APR DRG Description",
              "APR MDC Description", "APR Severity of Illness Description",
              "Payment Typology 2", "Payment Typology 3", "Birth Weight", "Total Charges", "Total Costs,"], axis=1)

age_group_index = {"0 to 17":1, "18 to 29":2, "30 to 49":3, "50 to 69":4, "70 or older":5}
gender_index = {"U":0, "F":1, "M":2}
risk_and_severity_index = {np.nan:0, "Minor":1, "Moderate":2, "Major":3, "Extreme":4}

df["Age Group"] = df["Age Group"].apply(lambda x: age_group_index[x])
df["Gender"] = df["Gender"].apply(lambda x: gender_index[x])
df["APR risk of Mortality"] = df["APR risk of Mortality"].apply(lambda x: risk_and_severity_index[x])

encoder = OrdinalEncoder()
df["Race"] = encoder.fit_transform(np.asarray(df["Race"]).reshape(-1, 1))
df["Ethnicity"] = encoder.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1, 1))
df["Type of Admission"] = encoder.fit_transform(np.asarray(df["Type of Admission"]).reshape(-1, 1))
df["CCSR Diagnosis Code"] = encoder.fit_transform(np.asarray(df["CCSR Diagnosis Code"]).reshape(-1, 1))
df["CCSR Procedure Code"] = encoder.fit_transform(np.asarray(df["CCSR Procedure Code"]).reshape(-1, 1))
df["APR Medical Surgical Description"] = encoder.fit_transform(np.asarray(df["APR Medical Surgical Description"]).reshape(-1, 1))
df["Payment Typology 1"] = encoder.fit_transform(np.asarray(df["Payment Typology 1"]).reshape(-1, 1))
df["Emergency Department Indicator"] = encoder.fit_transform(np.asarray(df["Emergency Department Indicator"]).reshape(-1, 1))

df.isna().sum()

df = df.drop("CCSR Procedure Code", axis=1)
df.dropna(subset=["Permanent Facility Id", "CCSR Diagnosis Code"])

X = df.drop(["Lenght of Stay"], axis=1)
y = df["Lenght of Stay"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtree = DecisionTreeRegressor(max_depth=10)
dtree.fit(X_train, y_train)
train_predictions = dtree.predict(X_train)
test_predictions = dtree.predict(X_test)

print("RMSE: Train ", np.sqrt(mean_squared_error(y_train, train_predictions)))
print("RMSE: Test ", np.sqrt(mean_squared_error(y_test, test_predictions)))


bins = [0, 5, 10, 20, 30, 50, 120]
labels = [5, 10, 20, 30, 50, 120]

df["los_bin"] = pd.cut(df["Lenght of Stay"], bins=bins)
df["los_label"] = pd.cut(df["Lenght of Stay"], bins=bins, labels=labels)
df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace(","," -"))
df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace("120","120+"))

f, ax = plt.subplots()
sns.countplot(x="los_bin", data=df)

new_X = df.drop(["Lenght of Stay", "los_bin", "los_label"], axis=1)
new_y = df["los_label"]

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=42)

dtree = DecisionTreeClassifier(max_depth=10)
dtree.fit(X_train, y_train)

train_predictions = dtree.predict(X_train)
test_predictions = dtree.predict(X_test)

print("Train Accuracy: ", accuracy_score(y_train, train_predictions))
print("Test Accuracy: ", accuracy_score(y_test, test_predictions))
print("Classification Report", classification_report(y_test, test_predictions))










#4
df = pd.read_csv("kidney_disease.csv")
df.drop("id", axis=1, inplace=True)

df["packed_cell_volume"] = pd.to_numeric(df["packed_cell_volume"], errors="coerce")
df["white_blood_cell_count"] = pd.to_numeric(df["white_blood_cell_count"], errors="coerce")
df["red_blood_cell_count"] = pd.to_numeric(df["red_blood_cell_count"], errors="coerce")

cat_cols = [col for col in df.columns if df[col].dtype == "object"]
num_cols = [col for col in df.columns if df[col].dtype == "object"]

for col in cat_cols:
    print(f"{col}: {df[col].unique()}")

df["diabetes_mellitus"].replace(to_replace={'\tno':"no", '\tyes':"yes", 'yes': "yes"}, inplace=True)
df["coronary_artery_disease"].replace(to_replace={'\tno':"no"}, inplace=True)
df["class"].replace(to_replace={'ckd\t':"ckd"}, inplace=True)

df["class"] = df["class"].map({"ckd":0, "notckd":1})

plt.figure(figsize=(15,15))
plotnumber = 1

for col in num_cols:
    if plotnumber <=14:
        ax = plt.subplots(3, 5, plotnumber)
        sns.distplot(df[col])
        plt.xlabel(col)

    plotnumber += 1

plt.tight_layout()
plt.show()

plt.figure()
sns.heatmap(df.corr(), annot=True, linecolor="white", linewidths=2)
plt.show()

def kde(col):
    grid = sns.FacetGrid(df, hue="class", height=6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()


kde("hemoglobin")
kde("white_blood_cell_count")
kde("packet_cell_volume")
kde("red_blood_cell_count")
kde("albumin")
kde("specific_gravity")

df.isna().sum().sort_values(ascending=False)

def solve_mv_random_value(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample


def solve_mv_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

solve_mv_random_value("red_blood_cells")
solve_mv_random_value("pus_cell")

for col in cat_cols:
    solve_mv_mode(col)

df[cat_cols].isna().sum()

for col in cat_cols:
    print(f"{col}: {df[col].nunique()}")

encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])


independent_col = [col for col in df.columns if col != "class"]
dependent_col = "class"

X = df[independent_col]
y = df[dependent_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print("Confusion Matrix: \n", cm)
print("Classification Report: \n", cr)


class_names = ["ckd", "notckd"]

plt.figure(figsize=(20, 10))

plot_tree(dtc, feature_names=independent_col, filled=True, rounded=True, fontsize=12)
plt.show()

feature_importance = pd.DataFrame({"feature": independent_col, "importance": dtc.feature_importances_})

print("Most importance feature: ", feature_importance.sort_values("importance", ascending=False).iloc[0])

plt.figure()
sns.barplot(x="importance", y="feature", data=feature_importance)
plt.title("Feature Importance")
plt.show()













#5

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop(["id"], axis=1)

plt.figure()
sns.countplot(x="stroke", data=df)
plt.title("Distribution of Stroke Class")
plt.show()

df.isnull().sum()

DT_bmi_pipe = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("dtr", DecisionTreeRegressor())
])

X = df[["gender", "age", "bmi"]].copy()

X.gender = X.gender.replace({"Male":0, "Female":1, "Other":-1}).astype(np.uint8)
missing = X[X.bmi.isna()]

X = X[X.bmi.isna()]
y = X.pop("bmi")

DT_bmi_pipe.fit(X, y)

prediction_bmi = pd.Series(DT_bmi_pipe.predict(missing[["gener", "age"]]), index=missing.index)
df.loc[missing.index, "bmi"] = prediction_bmi

df["gender"] = df["gender"].replace({"Male":0, "Female":1, "Other":-1}).astype(np.uint8)
df["Residence_type"] = df["Residence_type"].replace({"Rural":0, "Urban":1}).astype(np.uint8)
df["work_type"] = df["work_type"].replace({"private":0, "Self-employed":1, "Govt_job":2, "children":-1, "Never_worked":-2}).astype(np.uint8)

X = df[["gender", "age", "hypertension", "heart_disease", "work_type", "avg_glucose_level", "bmi"]]
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg_pipe = Pipeline(steps=[("scale", StandardScaler()), ("LR", LogisticRegression())])

logreg_pipe.fit(X_train, y_train)
y_pred = logreg_pipe.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))




