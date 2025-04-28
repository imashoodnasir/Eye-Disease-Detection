import tensorflow as tf
from ecsa_net_model import build_ecsa_net
from data_preprocessing import get_data_generators

# Paths
train_dir = 'path/to/train'
val_dir = 'path/to/val'
num_classes = 4  # Or 10 for second dataset

train_gen, val_gen = get_data_generators(train_dir, val_dir)

model = build_ecsa_net(num_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    tf.keras.callbacks.ModelCheckpoint('ecsa_net_best.h5', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=callbacks
)

model.save('ecsa_net_final.h5')
