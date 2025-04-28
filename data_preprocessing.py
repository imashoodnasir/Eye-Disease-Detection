import tensorflow as tf

def get_data_generators(train_dir, val_dir, image_size=(224, 224), batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator
