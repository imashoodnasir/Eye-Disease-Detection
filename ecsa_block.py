import tensorflow as tf
from attention_modules import channel_attention, spatial_attention

def ecsa_block(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x
