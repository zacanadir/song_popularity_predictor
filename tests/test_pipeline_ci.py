# test_pipeline_ci.py
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import train_m1
# ----------------------------
# Tiny dummy pipeline
# ----------------------------
numerical_cols = ['duration_ms', 'danceability', 'energy', 'loudness']
categorical_cols = ['explicit', 'key']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

dummy_model = DummyRegressor(strategy="mean")  # small placeholder model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', dummy_model)])

# Fit on a tiny dummy dataset
X_dummy = pd.DataFrame({
    'duration_ms': [200000, 180000],
    'danceability': [0.8, 0.6],
    'energy': [0.7, 0.5],
    'loudness': [-5.0, -7.0],
    'explicit': [True, False],
    'key': [5, 0]
})
y_dummy = pd.Series([50.0, 40.0])
pipeline.fit(X_dummy, y_dummy)

# ----------------------------
# Tests
# ----------------------------
def test_pipeline_predict_shape():
    """Predictions should match the number of input rows"""
    y_pred = pipeline.predict(X_dummy)
    assert y_pred.shape[0] == X_dummy.shape[0]

def test_pipeline_predict_numeric():
    """All predictions should be numeric"""
    y_pred = pipeline.predict(X_dummy)
    assert all(isinstance(p, (float, int)) for p in y_pred)

def test_pipeline_missing_column():
    """Pipeline should raise an error if a required column is missing"""
    X_missing_col = X_dummy.drop(columns=['energy'])
    with pytest.raises(ValueError):
        pipeline.predict(X_missing_col)

def test_pipeline_multiple_rows():
    """Test pipeline on multiple identical rows"""
    X_multi = pd.concat([X_dummy]*3, ignore_index=True)
    y_pred = pipeline.predict(X_multi)
    assert len(y_pred) == 6  # 2 rows * 3 repeats

def test_prepare_data():
    X_train,_,y_train,_ = train_m1.prepare_data()
    assert len(X_train) == len(y_train)