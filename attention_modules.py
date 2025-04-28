import tensorflow as tf
from tensorflow.keras import layers

def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1,1,channel))(avg_pool)
    
    dense_1 = layers.Conv2D(channel // ratio, (1,1), activation='relu')(avg_pool)
    dense_2 = layers.Conv2D(channel, (1,1), activation='sigmoid')(dense_1)
    
    scale = layers.multiply([input_feature, dense_2])
    return scale

def spatial_attention(input_feature):
    reduced = layers.Conv2D(input_feature.shape[-1]//8, (1,1), activation='relu')(input_feature)
    spatial = layers.Conv2D(1, (7,7), padding='same', activation='sigmoid')(reduced)
    
    scale = layers.multiply([input_feature, spatial])
    return scale
