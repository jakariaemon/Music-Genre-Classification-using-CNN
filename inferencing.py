import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU')
model = tf.keras.models.load_model('music_genre_classification_model.h5')

def preprocess_audio_clip(audio_path, target_size=(496, 369)):
    y, sr = librosa.load(audio_path)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(mels, ref=np.max)
    plt.figure(figsize=(target_size[1]/100, target_size[0]/100), dpi=100)  
    librosa.display.specshow(S_dB)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('temp_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    img = Image.open('temp_image.png').convert('RGB').resize((target_size[1], target_size[0]))  
    return np.array(img)

def predict_genre(audio_path):
    img_array = preprocess_audio_clip(audio_path)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    predictions = model.predict(img_array)
    return predictions[0]

def display_class_probabilities(predictions, genres):
    plt.figure(figsize=(10, 6))
    plt.bar(genres, predictions)
    plt.ylabel('Probability')
    plt.xlabel('Genre')
    plt.title('Music Genre Classification Probabilities')
    plt.show()

audio_path = 'rock.00008_clip_1.wav'  # Set your test audio. 3 seconds. 
genres = 'blues classical country disco pop hiphop metal reggae rock'.split()
predictions = predict_genre(audio_path)
display_class_probabilities(predictions, genres)
