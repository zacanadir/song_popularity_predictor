import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path

# ----------------------------
# Load data
# ----------------------------
def prepare_data(path="data/dataset.csv"):
    data = pd.read_csv(path, index_col=0)  # safer handling of unnamed index

    X = data.drop(columns=[
        "track_id", "artists", "album_name", "track_name",
        "track_genre", "popularity"
    ])
    y = data["popularity"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data()
# ----------------------------
# Train/test split
# ----------------------------


# ----------------------------
# Preprocessing
# ----------------------------
numerical_cols = [
    'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]
categorical_cols = ['explicit', 'key', 'mode', 'time_signature']

scaler = StandardScaler()
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_cols),
        ('cat', encoder, categorical_cols)
    ],
    verbose_feature_names_out=False
)

# ----------------------------
# Model + Pipeline
# ----------------------------
model = RandomForestRegressor(random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# ----------------------------
# Training
# ----------------------------
pipeline.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = pipeline.predict(X_test)
print("R² score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ----------------------------
# Save the trained pipeline
# ----------------------------
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, "models/trained_m1.joblib")
print("Model saved to models/trained_m1.joblib")
