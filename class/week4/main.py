   import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():

	# 1. Load the dataset (assuming 'titanic.csv' is available)
	# In a real project, you would download this from a source like Kaggle.
	# For simplicity, let's create a dummy DataFrame if the file isn't present
	try:
	    df = pd.read_csv('titanic.csv')
	except FileNotFoundError:
	    print("titanic.csv not found. Creating a dummy DataFrame for demonstration.")
	    data = {
		'PassengerId': range(1, 11),
		'Survived': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
		'Pclass': [3, 1, 3, 1, 3, 2, 3, 1, 2, 3],
		'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female'],
		'Age': [22, 38, 26, 35, 35, 28, 54, 2, 27, 4],
		'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
		'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
		'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 13.0, 51.86, 21.07, 13.0, 16.7],
		'Embarked': ['S', 'C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C']
	    }
	    df = pd.DataFrame(data)

	# 2. Preprocessing: Handle missing values and encode categorical features
	df['Age'].fillna(df['Age'].median(), inplace=True)
	df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
	df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
	df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

	# 3. Feature Selection: Define features (X) and target (y)
	features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
	X = df[features]
	y = df['Survived']

	# 4. Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# 5. Choose and train a model (Random Forest Classifier)
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)

	# 6. Make predictions on the test set
	y_pred = model.predict(X_test)

	# 7. Evaluate the model
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Model Accuracy: {accuracy:.2f}")

	# Example of a new prediction
	new_data = pd.DataFrame([[3, 0, 25, 0, 0, 7.0, 0, 1]], columns=features) # Male, 3rd class, 25, etc.
	prediction = model.predict(new_data)
	print(f"Prediction for new data (0=Not Survived, 1=Survived): {prediction[0]}")

if __name__ == "__main__":
    main()
