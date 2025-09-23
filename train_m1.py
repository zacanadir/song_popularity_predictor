import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("data/dataset.csv")


X = data.drop(columns=["Unnamed: 0", "track_id", "artists", "album_name", "track_name", "track_genre", "popularity"])
y = data["popularity"].apply(lambda x: float(x))


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

if __name__=="__main__":
    # Referencing numerical and categorical features
    numerical_cols= ['duration_ms', 'danceability', 'energy',  'loudness',
     'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo']
    categorical_cols=['explicit', 'key', 'mode', 'time_signature']

    # Instantiating preprocessing classes & specifying import args
    scaler = StandardScaler()
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    # Using ColumnTransformer to have one returned preprocessed DF
    preprocessor = ColumnTransformer(transformers=[('scl',scaler,numerical_cols),('enc',encoder,categorical_cols)],verbose_feature_names_out=False)

    # Models
    model = GradientBoostingRegressor()
    model2 = RandomForestRegressor()

    # Pipeline -> preprocessing + training
    pipeline = Pipeline(steps=[('pre',preprocessor),('mdl',model2)])
    pipeline.fit(X_train, y_train)
    model_score = pipeline.score(X_test, y_test)
    print(model_score)

    # Saving model
    joblib.dump(pipeline,"models/trained_m1.joblib")