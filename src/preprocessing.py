from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the raw dataset

BASE_DIR = Path(__file__).resolve().parent.parent

raw_file = BASE_DIR / "data" / "raw" / "california_housing.csv"
df = pd.read_csv(raw_file)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Clean the dataset

def clean_data(df):
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    return df

df = clean_data(df)

# Separate features and target

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Create preprocessing pipeline

preprocessor = Pipeline([
    ("scaler", StandardScaler())
])

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted pipeline
X_test_processed = preprocessor.transform(X_test)

print("Training data after preprocessing:", X_train_processed.shape)
print("Testing data after preprocessing:", X_test_processed.shape)

# Save processed training and testing data

processed_folder = BASE_DIR / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)

# Convert processed arrays into DataFrames
X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=X.columns
)

X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=X.columns
)

# Add target column
X_train_processed_df["MedHouseVal"] = y_train.values
X_test_processed_df["MedHouseVal"] = y_test.values

# Save the files
X_train_processed_df.to_csv(
    processed_folder / "train_processed.csv",
    index=False
)

X_test_processed_df.to_csv(
    processed_folder / "test_processed.csv",
    index=False
)

print("Processed datasets saved successfully!")

# Final verification

print("\nPreprocessing completed successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", X_train.shape[1])