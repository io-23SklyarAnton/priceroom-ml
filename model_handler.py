import joblib
import numpy as np
from keras.src.losses import MeanSquaredError
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

from db.crud.realty import get_realties
from patterns import SingletonMeta


class ModelHandler(metaclass=SingletonMeta):
    def __init__(self, model_path='models/realty_price_model.keras'):
        self.scaler = RobustScaler()
        self.model_path = model_path
        self.model = None
        self.column_features = None
        self.district_to_idx = None

    def load_data(self):

        realties = get_realties()

        structured_data = {
            "district_name_uk": [],
            "rooms_count": [],
            "total_square_meters": [],
            "price": []
        }

        for realty in realties:
            for key in structured_data:
                structured_data[key].append(getattr(realty, key))

        self.district_to_idx = {district: idx for idx, district in enumerate(set(structured_data["district_name_uk"]))}

        structured_data["district_name_uk"] = [self.district_to_idx[district] for district in
                                           structured_data["district_name_uk"]]

        self.column_features = ["rooms_count", "total_square_meters"]
        joblib.dump(self.column_features, 'models/column_features.pkl')

        return structured_data

    def build_model(self, input_shape, num_districts):

        district_input = Input(shape=(1,), name='district_input')
        district_embedding = Embedding(input_dim=num_districts, output_dim=5)(district_input)
        district_embedding_flat = Flatten()(district_embedding)

        other_features_input = Input(shape=(input_shape,), name='other_features_input')

        merged = Concatenate()([district_embedding_flat, other_features_input])

        x = Dense(256, activation='relu')(merged)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)

        output = Dense(1)(x)

        model = Model(inputs=[
            district_input, other_features_input
        ], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        return model

    def train_model(self):
        data = self.load_data()
        target = 'price'

        X = np.column_stack([data[feature] for feature in self.column_features])
        y = np.array(data[target])
        district_idxs = np.array(data['district_name_uk'])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_mae_scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            district_train, district_val = district_idxs[train_index], district_idxs[val_index]

            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_val = self.scaler.transform(X_val)

            model = self.build_model(
                X_train.shape[1], len(self.district_to_idx)
            )
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            model.fit(
                [district_train, X_train], y_train,
                epochs=50, batch_size=32,
                validation_data=([
                                     district_val, X_val
                                 ], y_val),
                callbacks=[early_stopping]
            )

            y_pred = model.predict([
                district_val, X_val
            ])
            fold_mae = mean_absolute_error(y_val, y_pred)
            fold_mae_scores.append(fold_mae)
            print(f"Fold MAE: {fold_mae:.2f}")

        avg_mae = np.mean(fold_mae_scores)
        print(f"Average MAE across folds: {avg_mae:.2f}")

        model.save(self.model_path)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.district_to_idx, 'models/district_to_idx.pkl')

    def prepare_features(self, district, rooms_count, total_square_meters):
        district_idx = self.district_to_idx[district]
        other_features = np.array([[rooms_count, total_square_meters]])  # Note the extra brackets
        other_features_scaled = self.scaler.transform(other_features)

        return [
            np.array([district_idx]),
            other_features_scaled
        ]

    def load_model(self):
        self.model = load_model(self.model_path)
        self.scaler = joblib.load('models/scaler.pkl')
        self.district_to_idx = joblib.load('models/district_to_idx.pkl')
        self.column_features = joblib.load('models/column_features.pkl')


if __name__ == '__main__':
    model_handler = ModelHandler()
    model_handler.train_model()
    model_handler.load_model()
    print("Model saved.")
