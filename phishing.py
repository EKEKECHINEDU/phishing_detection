# # Phishing Detection Machine Learning Project
# 
# _Ekeke Chinedu Christopher_

# #### Importing the necessary packages

# Data Manipulation 
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Customization
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# #### Loading the data

data = pd.read_csv("Phishing.csv")
print(data.head(5))

# remove identifier information 
data = data.drop(['id'], axis = 1)

# Examine the shape of the data
print(data.shape)

# #### Check for Missing Observations in data

print(data.isnull().sum())

# There are no missing observations in the data

# ### General Data Observations: Summary Statistics

print(data.describe())

# #### Structure of the target variable


# Classes to predict from.
print(data['CLASS_LABEL'].value_counts())

# #### Understanding the data types

print(data.info())

# #### Checking distribution of the features by target variable 

def summarize_by_class(data, group_col='CLASS_LABEL'):
    group = data.groupby(group_col)

    print("=== NoHttps Value Counts ===")
    print(group['NoHttps'].value_counts(), end="\n\n")

    print("=== UrlLength Mean ===")
    print(group['UrlLength'].mean(), end="\n\n")

    print("=== NumPercent Mean ===")
    print(group['NumPercent'].mean(), end="\n\n")

    print("=== NumAmpersand Mean ===")
    print(group['NumAmpersand'].mean(), end="\n\n")

    print("=== NumHash Mean ===")
    print(group['NumHash'].mean(), end="\n\n")

    print("=== IpAddress Value Counts ===")
    print(group['IpAddress'].value_counts(), end="\n\n")

summarize_by_class(data)

# #### Correlation of variables

plt.figure(figsize=(30, 30))
sns.heatmap(data.corr(),annot=True,cmap='viridis',linewidths=.5)
plt.show()


# ## Visualization

# #### Bar plot of 3 binary variables by Target Variable 

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

binary_vars = ['InsecureForms', 'SubmitInfoToEmail', 'IframeOrFrame']
for i, var in enumerate(binary_vars):
    sns.countplot(data=data, x=var, hue='CLASS_LABEL', ax=axes[i])
    axes[i].set_title(f'Bar Plot of {var} by Phishing Site')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Count')
    axes[i].legend(title="CLASS LABEL")

plt.tight_layout()
plt.show()


# #### Density Plot of chosen continuous variables by target variable

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

float_vars = ['PctExtHyperlinks', 'PctExtResourceUrls', 'PctNullSelfRedirectHyperlinks']
for i, var in enumerate(float_vars):
    sns.kdeplot(data=data, x=var, hue='CLASS_LABEL', fill=True, ax=axes[i])
    axes[i].set_title(f'Density Plot of {var} by Phishing Site')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.show()

# ## Preprocessing & Feature Engineering

y = data['CLASS_LABEL']
X = data.drop(['CLASS_LABEL'], axis = 1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Feature Selection

# #### Feature Selection by RFE

#Scale the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Apply RFE with logistic regression
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=15)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)

print("Selected features based on Lasso:", X_train.columns[rfe.get_support()])


# #### Feature Importance by Random Forest

# Train random forest and get feature importances
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scaled, y_train)
importances = model_rf.feature_importances_

# Convert to Pandas Series
feature_importances = pd.Series(importances, index=X_train.columns)

# Select the top 15 features
top_features = feature_importances.nlargest(15)

# Print the top 15 features
print(top_features)

# Store them in a list
top_feature_list = top_features.index.tolist()

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='skyblue')
plt.title("Top 15 Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Print the stored list of top features
print("Top 15 Features:", top_feature_list)


# #### Feature Selection by Lasso

# Train Lasso model
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso.fit(X_train_scaled, y_train)

# Get important features
lasso_features = X_train.columns[np.abs(lasso.coef_).ravel() > 0].tolist()

# Get feature coefficients
lasso_coefs = np.abs(lasso.coef_).ravel()

# Convert to Pandas Series
lasso_features = pd.Series(lasso_coefs, index=X_train.columns)

# Select the top 15 features
top_lasso_features = lasso_features.nlargest(15)

# Plot the top 15 features
plt.figure(figsize=(6, 8))
top_lasso_features.sort_values().plot(kind='barh', color='dodgerblue')
plt.title("Top 15 Features (Lasso Regression)")
plt.xlabel("Coefficient Magnitude")
plt.ylabel("Features")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Print the selected features
top_lasso_list = top_lasso_features.index.tolist()
print("Top 15 Features (Lasso Regression):", top_lasso_list)


# #### Feature Selection by Mutual Information

# Compute mutual information
mi_scores = mutual_info_classif(X_train_scaled, y_train)
mi_features = pd.Series(mi_scores, index=X_train.columns).nlargest(15)

print(mi_features)


# Select the top 15 features
top_mi_features = mi_features.nlargest(15)

# Plot the top 15 features
plt.figure(figsize=(6, 8))
top_mi_features.sort_values().plot(kind='barh', color='coral')
plt.title("Top 15 Features (Mutual Information)")
plt.xlabel("Mutual Information Score")
plt.ylabel("Features")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Print the selected features
top_mi_list = top_mi_features.index.tolist()
print("Top 15 Features (Mutual Information):", top_mi_list)


# Plot Mutual Information Features 
def plot_mutual_information(X, y, top_n=15):
    """
    Compute and plot the top N features based on mutual information.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        top_n (int): Number of top features to display.
    """
    # Compute Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Select top N features
    top_mi = mi_series.head(top_n)
    
    # Save top feature names for further use
    top_mi_list = top_mi.index.tolist()

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_mi, y=top_mi.index, palette="coolwarm", edgecolor="black")

    # Labels and aesthetics
    plt.xlabel("Mutual Information Score", fontsize=12, labelpad=10)
    plt.ylabel("Features", fontsize=12, labelpad=10)
    plt.title(f"Top {top_n} Features Based on Mutual Information", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Show values on bars
    for index, value in enumerate(top_mi):
        plt.text(value + 0.01, index, f"{value:.3f}", va='center', fontsize=10, color="black")

    # Improve layout
    plt.tight_layout()
    plt.show()
    
    return top_mi_list

# Alternative Plot
top_mi_list = plot_mutual_information(X_train, y_train)
print("Top MI Features:", top_mi_list)


# #### Selected Features based on all selection criteria

subset_fishing = data[['NumDots', 'PathLevel', 'NumDash', 'NumSensitiveWords', 'PctExtHyperlinks',
       'PctExtResourceUrls', 'InsecureForms', 'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
       'SubmitInfoToEmail', 'IframeOrFrame', 'CLASS_LABEL']]

print(subset_fishing.head())  


## Obtain summary statistics of just the selected variables
# Create the summary DataFrame
summary_df = pd.DataFrame({
    'Mean': round(subset_fishing.mean(), 4),
    'Std' : round(subset_fishing.std(), 4),
    'Min': round(subset_fishing.min(numeric_only=True)),  # Min value (only numeric columns) 
    'Median': round(subset_fishing.median(), 2),
    'Max': round(subset_fishing.max(numeric_only=True), 0),  # Max value (only numeric columns)
    'Unique Values': subset_fishing.nunique(),     # Number of unique values
    'Data Type': subset_fishing.dtypes                 # Data type of each column
})

# Reset index for better display
summary_df = summary_df.reset_index().rename(columns={'index': 'Column'})

# Display summary
print(summary_df)


# ## Machine Learning Classification

# Split Train and Validation based on selected variables
y = subset_fishing['CLASS_LABEL']
X = subset_fishing.drop(['CLASS_LABEL'], axis = 1)

Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state = 42)

#Ascertain class distribution of target variable in train and validation sets
# Set figure size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for y_train
sns.countplot(x=ytrain, ax=axes[0], palette="Blues")
axes[0].set_title("Distribution of Training target variable")
axes[0].set_xlabel("Class Label")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, 5000)  # Set y-axis limit

# Plot for y_test
sns.countplot(x=yval, ax=axes[1], palette="Oranges")
axes[1].set_title("Distribution of Validation target variable")
axes[1].set_xlabel("Class Label")
axes[1].set_ylabel("Count")
axes[1].set_ylim(0, 5000)  # Set same y-axis limit

# Show plot
plt.tight_layout()
plt.show()


# Function to train classifiers and compute performance metrics
def evaluate_classifiers(X, y, test_size=0.2, random_state=42):
    """
    Train multiple classifiers and compute Accuracy, Precision, Recall, and F1-score.
    
    Parameters:
        X (DataFrame or array): Feature matrix.
        y (Series or array): Target variable.
        test_size (float): Test split ratio.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        DataFrame: Performance metrics for each classifier.
    """
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize features (important for SVM and KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC()
    }
    
    # Store results
    results = []

    for name, clf in classifiers.items():
        # Use scaled data for classifiers that require normalization
        if name in ["Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression", "Linear Discriminant Analysis"]:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        results.append([name, accuracy, precision, recall, f1])
    
    # Convert results to DataFrame
    metrics_df = pd.DataFrame(results, columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-score"])
    
    return metrics_df


metrics_df = evaluate_classifiers(Xval, yval)
print(metrics_df)


## Train Random forest Classifier
random_model = RandomForestClassifier(n_estimators=250, n_jobs = -1, random_state=42)

random_model.fit(Xtrain, ytrain)

y_pred = random_model.predict(Xval)

#Checking training accuracy
random_model_accuracy = round(random_model.score(Xtrain, ytrain)*100,2)
print(round(random_model_accuracy, 2), '%')


#Checking validation accuracy
random_val_accuracy = round(random_model.score(Xval, yval)*100,2)
print(round(random_val_accuracy, 2), '%')
