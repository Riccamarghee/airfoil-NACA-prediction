import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow.python.framework import ops
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import sklearn.utils
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tensorflow.keras import layers, regularizers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import ExponentialDecay

keras.utils.set_random_seed(seed=42)

def convert_multivariate_label(labels):
    
    new_labels = []
    for label in labels:
        # Primi due numeri (intensità del difetto sopra e sotto)
        intensity_above = int(label[0])
        intensity_below = int(label[1])

        # Terzo e quarto numero (bump o scavo sopra/sotto)
        bump_above = int(label[2])
        bump_below = int(label[3])

        # Conversione dei difetti sopra e sotto da 0-1-2 a -2-2
        if bump_above == 1:
            intensity_above = intensity_above  # Mantieni il bump come positivo
        elif bump_above == 0:
            intensity_above = -intensity_above  # Converti lo scavo in negativo

        if bump_below == 1:
            intensity_below = intensity_below  # Mantieni il bump come positivo
        elif bump_below == 0:
            intensity_below = -intensity_below  # Converti lo scavo in negativo

        # Quinto numero (taglio del trailing edge)
        trailing_edge_cut = int(label[4])

        
        new_labels.append([intensity_above, intensity_below, trailing_edge_cut])

    return np.array(new_labels)

def shuffle_points(points, random_state=42):
    
    shuffled_points = points.copy()
    for i in range(shuffled_points.shape[0]):  # Itera su ogni esempio del dataset
        np.random.shuffle(shuffled_points[i], random_state)  # Shuffle lungo la dimensione dei punti (5)
    return shuffled_points

def convert_naca_label(labels):
    
    new_labels = []
    for label in labels:
       
        first = int(label[0])
        second = int(label[1])
        third = int(label[2:4])

        # Salva il nuovo label con solo i primi due valori convertiti e il quinto
        new_labels.append([first, second, third])

    return np.array(new_labels)





def augment_data(features, labels, noise_factor=0.05, augment_factor=2):
    
    augment_x = []
    augment_y = []
    #augment_naca = []

    for _ in range(augment_factor):
        perturbed_features = features.copy()

        # Perturba sia le coordinate (prime 3 colonne) che le feature (dalla quarta colonna in poi)
        perturbed_features += noise_factor * np.random.randn(*features.shape)

        augment_x.append(perturbed_features)
        augment_y.append(labels)
        #augment_naca.append(naca_codes)

    # Stack dei dati augmentati
    augment_x = np.vstack(augment_x)
    augment_y = np.vstack(augment_y)
    #augment_naca = np.hstack(augment_naca)

    return augment_x, augment_y



def load_and_preprocess_naca_data(points_file, labels_file, batch_size, train_size, test_size, augment=False, augment_factor=2 ,scale= True, only_p= False):
    """
    Carica e pre-processa i dati per la regressione multivariata.
    """
    # Carica i dati dai file .npy
    points = np.load(points_file)
    labels = np.load(labels_file)
    #points = points[:,0:8, 1:2]



    labels = convert_naca_label(labels)

    NUM_POINTS = points.shape[1]
    NUM_FEATURES = points.shape[2]   # Aggiungiamo una colonna vuota come fatto in precedenza

    

    if augment:
        augment_x, augment_y= augment_data(points, labels, noise_factor=0.05, augment_factor=augment_factor)
        points = np.vstack([points, augment_x])
        labels = np.vstack([labels, augment_y])


    scalers = []
    if scale:
        
        for i in range(NUM_FEATURES):
            scaler = StandardScaler()
            
            # Prendi tutte le osservazioni di una singola feature i (asse 2)
            feature_data = points[:, :, i].reshape(-1, 1)
            
            # Standardizza la feature i
            standardized_feature = scaler.fit_transform(feature_data)
            
            # Ricostruisci il tensore originale con la feature trasformata
            points[:, :, i] = standardized_feature.reshape(points[:, :, i].shape)
            
            scalers.append(scaler)  

    if only_p:
        points = points[:,:,1:2]
        NUM_FEATURES = points.shape[2] 

    #points_scaled = points.copy()

    
    points, labels = sklearn.utils.shuffle(points, labels, random_state=42)

   
    x_train_val, x_test, y_train_val, y_test= train_test_split(
        points, labels, test_size=test_size, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val,  train_size=train_size, random_state=42)

   
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return points, labels, train_ds, val_ds, test_ds, NUM_POINTS, NUM_FEATURES




class SimpleBaseModel:
    def __init__(self, num_points, num_features, num_targets):
        """
        Initialize the model with given parameters.
        """
        self.num_points = num_points
        self.num_features = num_features
        self.num_targets = num_targets
        self.model = self._build_model()

    def _build_model(self):
        
        inputs = layers.Input(shape=(self.num_points, self.num_features))
        x = layers.Flatten()(inputs)

        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dense(8, activation='relu')(x)

        # Output layer
        outputs = layers.Dense(self.num_targets, activation='linear')(x)

       
        model = models.Model(inputs=inputs, outputs=outputs)
        
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=150,
            decay_rate=0.985,
            staircase=True  
        )
        model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                        loss='mean_squared_error', metrics=['mse']) 
        return model

    def train(self, train_ds, val_ds, epochs=1000, patience=10):
        """
        Train the model with given training and validation datasets.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[early_stopping]
        )
        return history

    def evaluate(self, test_ds):
        
        test_loss, test_mae = self.model.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        return test_loss, test_mae

    def plot_training_history(self, history):
        """
        Plot the training and validation loss over epochs.
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(12, 6))
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Andamento della Loss')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def show_predictions(self, test_ds, ax, title="Model Predictions", model_name="Model"):
        
     
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        y_pred = self.model.predict(test_ds)

  
        first_digit_true = y_true[:, 0]
        second_digit_true = y_true[:, 1]
        third_digit_true = y_true[:, 2]

        first_digit_pred = y_pred[:, 0]
        second_digit_pred = y_pred[:, 1]
        third_digit_pred = y_pred[:, 2]


        digits = [
            ("First Digit (0-9)", first_digit_true, first_digit_pred),
            ("Second Digit (0-9)", second_digit_true, second_digit_pred),
            ("Third Digit (5-50)", third_digit_true, third_digit_pred),
        ]

        # Plot each digit
        for i, (digit_name, y_true_digit, y_pred_digit) in enumerate(digits):
            scatter = ax[i].scatter(
                y_true_digit,
                y_pred_digit,
                edgecolors="k",
                alpha=0.7
            )
            ax[i].plot(
                [y_true_digit.min(), y_true_digit.max()],
                [y_true_digit.min(), y_true_digit.max()],
                "k--",
                lw=2
            )
            ax[i].set_title(f"{model_name} - {digit_name}")
            ax[i].set_xlabel("True Values")
            ax[i].set_ylabel("Predicted Values")
            ax[i].grid()

    

    def evaluate_relative_error(self, test_ds):
        """
        Calculate relative errors for the model's predictions.
        """
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        y_pred = self.model.predict(test_ds)

        errors = np.abs(y_true - y_pred)
        mae = np.mean(errors)
        max_error = np.max(errors)
        max_error_position = np.unravel_index(np.argmax(errors), errors.shape)

        mae_1 = np.mean(errors[:, 0])
        mae_2 = np.mean(errors[:, 1])
        mae_3 = np.mean(errors[:, 2])

        print(f"Errore Assoluto Medio sulla I cifra (MAE): {mae_1:.4f}, {mae_1 / 9 * 100:.2f}%")
        print(f"Errore Assoluto Medio sulla II cifra (MAE): {mae_2:.4f}, {mae_2 / 9 * 100:.4f}%")
        print(f"Errore Assoluto Medio sulla III cifra (MAE): {mae_3:.4f}, {mae_3 / 45 * 100:.4f}%")
        print(f"Errore Medio assoluto (MAE): {mae:.4f}")
        print(f"Errore Massimo: {max_error:.4f} in posizione {max_error_position}")

        return mae, max_error, max_error_position
    


    def evaluate_accuracy(self, test_ds):
        """
        Calculate classification accuracy by rounding predictions and comparing with true values.
        """
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        y_pred = self.model.predict(test_ds)

       
        y_pred_rounded = np.round(y_pred)

       
        accuracy = np.mean(np.all(y_true == y_pred_rounded, axis=1))
        correct_predictions = np.sum(np.all(y_true == y_pred_rounded, axis=1))

        print(f"Correct Predictions: {correct_predictions}/{y_true.shape[0]}")
        print(f"Test Accuracy (Classification): {accuracy:.4f}")

        return accuracy, correct_predictions
    
    def summary(self):
        self.model.summary()
        return



class PointTransformerLayer(layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super(PointTransformerLayer, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.mlp_q = layers.Dense(out_dim)
        self.mlp_k = layers.Dense(out_dim)
        self.mlp_v = layers.Dense(out_dim)
        self.mlp_gamma = layers.Dense(out_dim, activation="sigmoid")

    def call(self, x):
        q = self.mlp_q(x)
        k = self.mlp_k(x)
        v = self.mlp_v(x)

       
        attn = tf.nn.softmax(tf.einsum("bpi,bqi->bpq", q, k), axis=-1)

        
        aggregated = tf.einsum("bpq,bqj->bpj", attn, v)

        
        out = self.mlp_gamma(x) * aggregated
        return out


class PointTransformerModel:
    def __init__(self, num_points, num_features, num_targets, invariant = False):
        
        self.num_points = num_points
        self.num_features = num_features
        self.num_targets = num_targets
        self.invariant = invariant
        self.model = self._build_model()

    def _build_model(self):
        
        inputs = layers.Input(shape=(self.num_points, self.num_features))

        # Transformer Layers
        x = PointTransformerLayer(32)(inputs)
        x = PointTransformerLayer(64)(inputs)
        

        # Global Pooling (symmetry-preserving)
        if self.invariant:
            x = layers.GlobalMaxPooling1D()(x)
        else:
            x = layers.Flatten()(x)

        
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(self.num_targets, activation="linear")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=["mse"])
        return model

    def train(self, train_ds, val_ds, epochs=1000, patience=10):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[early_stopping],
        )
        return history

    def evaluate(self, test_ds):
        
        test_loss, test_mae = self.model.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        return test_loss, test_mae

    def plot_training_history(self, history):
        
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        plt.figure(figsize=(12, 6))
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate_relative_error_naca(self, test_ds):
        
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        y_pred = self.model.predict(test_ds)

        errors = np.abs(y_true - y_pred)
        mae = np.mean(errors)
        max_error = np.max(errors)
        max_error_position = np.unravel_index(np.argmax(errors), errors.shape)

        mae_1 = np.mean(errors[:, 0])
        mae_2 = np.mean(errors[:, 1])
        mae_3 = np.mean(errors[:, 2])

        print(f"Errore Assoluto Medio sulla I cifra (MAE): {mae_1:.4f}, {mae_1 / 10 * 100:.4f}%")
        print(f"Errore Assoluto Medio sulla II cifra (MAE): {mae_2:.4f}, {mae_2 / 10 * 100:.4f}%")
        print(f"Errore Assoluto Medio sulla III cifra (MAE): {mae_3:.4f}, {mae_3 / 45 * 100:.4f}%")
        print(f"Errore Medio assoluto (MAE): {mae:.4f}")
        print(f"Errore Massimo: {max_error:.4f} in posizione {max_error_position}")

        return mae, max_error, max_error_position

    def evaluate_accuracy(self, test_ds):
        
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        y_pred = self.model.predict(test_ds)

        # Round predictions to the nearest integer
        y_pred_rounded = np.round(y_pred)

        # Calculate accuracy
        accuracy = np.mean(np.all(y_true == y_pred_rounded, axis=1))
        correct_predictions = np.sum(np.all(y_true == y_pred_rounded, axis=1))

        print(f"Correct Predictions: {correct_predictions}/{y_true.shape[0]}")
        print(f"Test Accuracy (Classification): {accuracy:.4f}")

        return accuracy, correct_predictions
    
    def show_predictions_naca(self, test_ds, ax, title="Model Predictions", model_name="Model"):
       
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
        y_pred = self.model.predict(test_ds)

        # Extract digits
        first_digit_true = y_true[:, 0]
        second_digit_true = y_true[:, 1]
        third_digit_true = y_true[:, 2]

        first_digit_pred = y_pred[:, 0]
        second_digit_pred = y_pred[:, 1]
        third_digit_pred = y_pred[:, 2]

        # Define digits for plotting
        digits = [
            ("First Digit (-2:2)", first_digit_true, first_digit_pred),
            ("Second Digit (-2:2)", second_digit_true, second_digit_pred),
            ("Third Digit (0:2)", third_digit_true, third_digit_pred),
        ]

        # Plot each digit
        for i, (digit_name, y_true_digit, y_pred_digit) in enumerate(digits):
            scatter = ax[i].scatter(
                y_true_digit,
                y_pred_digit,
                edgecolors="k",
                alpha=0.7
            )
            ax[i].plot(
                [y_true_digit.min(), y_true_digit.max()],
                [y_true_digit.min(), y_true_digit.max()],
                "k--",
                lw=2
            )
            ax[i].set_title(f"{model_name} - {digit_name}")
            ax[i].set_xlabel("True Values")
            ax[i].set_ylabel("Predicted Values")
            ax[i].grid()
            
    def summary(self):
        self.model.summary()
        return


from sklearn.model_selection import KFold

# K-Fold Cross-Validation
def kfold_cross_validation_MLP(data, labels, num_points, num_features, num_targets, n_splits=5, batch_size=32, epochs=100, patience=10):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    all_test_maes = []

    for train_index, test_index in kf.split(data):
        print(f"Fold {fold}/{n_splits}")
        
        # Split data
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

        # Further split training data into training and validation sets
        val_split = int(0.2 * len(train_data))
        val_data, val_labels = train_data[:val_split], train_labels[:val_split]
        train_data, train_labels = train_data[val_split:], train_labels[val_split:]
        val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(batch_size)

        # Initialize and train model
        model = SimpleBaseModel(num_points, num_features, num_targets)
        model.train(train_ds, val_ds, epochs=epochs, patience=patience)

        # Evaluate the model
        test_loss, test_mae = model.evaluate(test_ds)
        all_test_maes.append(test_mae)
        fold += 1

    # Output mean and std deviation of test MAEs
    print("\nCross-Validation Results:")
    print(f"Mean Test MAE: {np.mean(all_test_maes):.4f}")
    print(f"Std Test MAE: {np.std(all_test_maes):.4f}")

    return np.mean(all_test_maes), np.std(all_test_maes)

def kfold_cross_validation_PT(data, labels, num_points, num_features, num_targets, n_splits=5, batch_size=32, epochs=100, patience=10):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    all_test_maes = []

    for train_index, test_index in kf.split(data):
        print(f"Fold {fold}/{n_splits}")
        
        # Split data
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)

        # Further split training data into training and validation sets
        val_split = int(0.2 * len(train_data))
        val_data, val_labels = train_data[:val_split], train_labels[:val_split]
        train_data, train_labels = train_data[val_split:], train_labels[val_split:]
        val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(batch_size)

        # Initialize and train model
        model = PointTransformerModel(num_points, num_features, num_targets)
        model.train(train_ds, val_ds, epochs=epochs, patience=patience)

        # Evaluate the model
        test_loss, test_mae = model.evaluate(test_ds)
        all_test_maes.append(test_mae)
        fold += 1

    # Output mean and std deviation of test MAEs
    print("\nCross-Validation Results:")
    print(f"Mean Test MAE: {np.mean(all_test_maes):.4f}")
    print(f"Std Test MAE: {np.std(all_test_maes):.4f}")

    return np.mean(all_test_maes), np.std(all_test_maes)



