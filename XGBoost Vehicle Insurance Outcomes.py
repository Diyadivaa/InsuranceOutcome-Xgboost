import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GridSearchCV

# Load the data set into a Pandas dataframe
df = pd.read_csv("Car_Insurance_Claim.csv")

# Handle missing values
df.dropna(inplace=True)

age_counts = df['AGE'].value_counts()

print(age_counts)


# Use the groupby function to split the data into groups
groups = df.groupby('GENDER')

# Iterate over the groups and plot a boxplot for each group
for name, group in groups:
    plt.boxplot(group['ANNUAL_MILEAGE'])
    plt.title(name)
    plt.show()

# Use the groupby function to split the data into groups
groups = df.groupby('VEHICLE_TYPE')

# Iterate over the groups and plot a boxplot for each group
for name, group in groups:
    plt.boxplot(group['ANNUAL_MILEAGE'])
    plt.title(name)
    plt.show()


value_mapping = {
    "AGE": {
        "16-25": 0,
        "26-39": 1,
        "40-64": 2,
        "65+": 3
    },
    "GENDER": {
        "male": 0,
        "female": 1
    },
    "RACE": {
        "majority": 0,
        "minority": 1

    },
    "DRIVING_EXPERIENCE": {
        "0-9y": 0,
        "10-19y": 1,
        "20-29y": 2,
        "30y+" : 3
    },
    "EDUCATION": {
        "high school": 0,
        "none": 1,
        "university": 2
    },
    "INCOME": {
        "upper class": 0,
        "working class": 1,
        "middle class": 2,
        "poverty": 3 
    },
    "VEHICLE_YEAR": {
        "after 2015": 0,
        "before 2015": 1
    },
    "VEHICLE_TYPE": {
        "sedan": 0,
        "sports car": 1
    }
}


# Use the mapping to convert the string columns to numerical values
df[list(value_mapping.keys())] = df[list(value_mapping.keys())].apply(lambda x: x.map(value_mapping[x.name]))

# viewing dataframe
df.head(1)

#dropping ID
#Important note in Real world cases we may end up dropping the postal code race information and gender. It depends on the context, but using this can lead to issues with discrimination.
df = df.drop(df.columns[0], axis=1)


# Calculate the Pearson correlation coefficient between 'age' and 'outcome'
r, _ = stats.pearsonr(df['AGE'], df['OUTCOME'])

# Create the correlation plot
plt.scatter(df['AGE'], df['OUTCOME'])
plt.title(f'Correlation between Age and Outcome (r = {r:.2f})')


# Split the data into training and test sets
X = df.drop("OUTCOME", axis=1)
y = df["OUTCOME"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# The range of values to search for the learning rate and number of estimators
param_grid = {'learning_rate': [0.1, 0.5, 1.0],
              'n_estimators': [50, 100, 200]}

# Create the XGBClassifier model
model = xgb.XGBClassifier(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print the best combination of hyperparameters
print(grid_search.best_params_)

# Train model based on learning_rate:.1 and n_estimators = 50.
# we can narrow this down, but for now we'll use this to prevent overfitting.
model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.2f}")