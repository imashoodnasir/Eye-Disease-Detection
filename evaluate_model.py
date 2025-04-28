import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_preprocessing import get_data_generators

# Load saved model
model = tf.keras.models.load_model('ecsa_net_best.h5', compile=False)

# Load validation data
train_dir = 'path/to/train'
val_dir = 'path/to/val'
train_gen, val_gen = get_data_generators(train_dir, val_dir)

# Predict
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_gen.classes

# Classification report
print(classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys()))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_gen.class_indices.keys())
disp.plot(cmap='Blues')
plt.show()
