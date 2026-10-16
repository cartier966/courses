import numpy as np
import pandas as pd
from pathlib import Path


DATA_FOLDER = Path(__file__).parent


def load_customer_churn_data():
	"""Load the customer churn practice data."""
	return pd.read_csv(DATA_FOLDER / "customer_churn.csv")


def load_house_prices_data():
	"""Load the house prices practice data."""
	return pd.read_csv(DATA_FOLDER / "house_prices.csv")


def load_student_performance_data():
	"""Load the student performance practice data."""
	return pd.read_csv(DATA_FOLDER / "student_performance.csv")


# NumPy arrays with one, two, and three dimensions.
one_dimensional = np.array([10, 20, 30, 40])
two_dimensional = np.array([[1, 2, 3], [4, 5, 6]])
three_dimensional = np.array(
	[
		[[1, 2], [3, 4]],
		[[5, 6], [7, 8]],
	]
)

# A small table of sample data for pandas exercises.
students = pd.DataFrame(
	{
		"name": ["Ava", "Ben", "Cara"],
		"age": [20, 21, 19],
		"score": [88, 92, 95],
	}
)


# print(f"NumPy: {np.__version__}")
# print(f"pandas: {pd.__version__}")
# print("1D array:", one_dimensional)
# print("2D array:\n", two_dimensional)
# print("3D array shape:", three_dimensional.shape)
# print("Student data:\n", students)

customer_churn = load_customer_churn_data()
house_prices = load_house_prices_data()
student_performance = load_student_performance_data()

# print("Customer churn rows:", len(customer_churn))
# print("House prices rows:", len(house_prices))
# print("Student performance rows:", len(student_performance))


# --- NumPy array manipulation ---

# print("\nSliced 1D array (index 1 to 3):", one_dimensional[1:3])
# print("2D array column 1:", two_dimensional[:, 1])
# print("2D array transposed:\n", two_dimensional.T)
# print("3D array flattened:", three_dimensional.flatten())
# print("Element-wise +10:", one_dimensional + 10)
# print("Mean/std of 1D array:", one_dimensional.mean(), one_dimensional.std())
# print("Row sums of 2D array:", two_dimensional.sum(axis=1))
# print("Reshaped 1D array to 2x2:\n", one_dimensional.reshape(2, 2))



def clean_customer_churn(df):
	"""Clean messy values in the customer churn dataset."""
	# print(df.info())
	# print(df.head())
	# print(df.isna().sum())
	df = df.drop_duplicates().copy()
	df["monthly_spend"] = pd.to_numeric(
		df["monthly_spend"].astype(str).str.strip(), errors="coerce"
	)
	df["monthly_spend"] = df["monthly_spend"].fillna(df["monthly_spend"].mean())
	df["age"] = df["age"].fillna(df["age"].median())
	df["months_active"] = df["months_active"].fillna(df["months_active"].median())
	df["contract"] = df["contract"].str.strip().str.lower().str.replace(" ", "-")
	df["city"] = df["city"].fillna("Unknown").str.strip().str.title()
	for col in ("has_support", "churned"):
		df[col] = df[col].str.strip().str.lower().map({"yes": True, "no": False})
	return df


def clean_house_prices(df):
	"""Clean messy values in the house prices dataset."""
	# print(df.info())
	# print(df.head())
	# print(df.isna().sum())
	df = df.drop_duplicates().copy()
	df["price"] = (
		df["price"].astype(str).str.replace(r"[$,]", "", regex=True)
	)
	df["price"] = pd.to_numeric(df["price"], errors="coerce")
	df["price"] = df["price"].fillna(df["price"].median())
	df["square_feet"] = pd.to_numeric(
		df["square_feet"].astype(str).str.strip(), errors="coerce"
	)
	df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
	df["year_built"] = pd.to_numeric(df["year_built"], errors="coerce")
	df["year_built"] = df["year_built"].fillna(df["year_built"].median()).astype(int)
	df["neighborhood"] = df["neighborhood"].fillna("Unknown").str.strip().str.title()
	return df


def clean_student_performance(df):
	"""Clean messy values in the student performance dataset."""

	df = df.drop_duplicates().copy()
	print(df)
	input("Press Enter to continue...")

	df["study_hours"] = pd.to_numeric(
		df["study_hours"], errors="coerce"
	)

	print(df)
	input("Press Enter to continue...")
	

	df["study_hours"] = df["study_hours"].fillna(df["study_hours"].median())
	print(df)
	input("Press Enter to continue...")

	# df["attendance_percent"] = df["attendance_percent"].astype(float)
	df["attendance_percent"] = pd.to_numeric(
		df["attendance_percent"].astype(str).str.rstrip("%"), errors="coerce"
	)
	print(df)
	input("Press Enter to continue...")
		

	df["attendance_percent"] = df["attendance_percent"].clip(upper=100, lower=0)
	print(df)
	input("Press Enter to continue...")
		

	df["attendance_percent"] = df["attendance_percent"].fillna(
		df["attendance_percent"].mean()
	)
	print(df)
	input("Press Enter to continue...")

	df["practice_tests"] = pd.to_numeric(df["practice_tests"], errors="coerce").fillna(0)
	print(df)
	input("Press Enter to continue...")
		

	df["subject"] = df["subject"].str.strip().str.title()
	print(df)

	input("Press Enter to continue...")
		
	df["passed"] = df["passed"].str.strip().str.lower().map({"yes": True, "no": False})

	print(df)
	input("Press Enter to continue...")
		
	return df


customer_churn_clean = clean_customer_churn(customer_churn)
house_prices_clean = clean_house_prices(house_prices)
student_performance_clean = clean_student_performance(student_performance)

#print("\nCleaned customer churn:\n", customer_churn_clean)
#print("\nCleaned house prices:\n", house_prices_clean)
print("\nCleaned student performance:\n", student_performance_clean)

#print("\nAverage monthly spend by contract type:\n",
#	customer_churn_clean.groupby("contract")["monthly_spend"].mean())
#print("\nAverage price by neighborhood:\n",
#	house_prices_clean.groupby("neighborhood")["price"].mean())
#print("\nPass rate by subject:\n",
#	student_performance_clean.groupby("subject")["passed"].mean())


