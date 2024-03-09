import os
import random
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
import librosa.display 
import shutil

def create_directories(base_path, genres):
    """
    Create directories for storing 3-second audio clips and their spectrograms within a dataset folder.
    """
    paths = ['dataset/audio3sec', 'dataset/spectrograms/train', 'dataset/spectrograms/test']
    for path in paths:
        for genre in genres:
            dir_path = os.path.join(base_path, path, genre)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

def split_save_audio_files(base_path, genres, clip_duration=3):
    """
    Split each audio file into smaller clips and save them.
    """
    source_path = os.path.join(base_path)
    for genre in genres:
        genre_path = os.path.join(source_path, genre)
        for filename in os.listdir(genre_path):
            song_path = os.path.join(genre_path, filename)
            audio = AudioSegment.from_file(song_path)
            for clip in range(0, 10):  # Assuming each file is exactly 30 seconds
                start = clip * clip_duration * 1000
                end = (clip + 1) * clip_duration * 1000
                clip_audio = audio[start:end]
                clip_path = os.path.join(base_path, 'dataset/audio3sec', genre, f"{filename[:-4]}_clip_{clip}.wav")
                clip_audio.export(clip_path, format="wav")

def generate_spectrograms(base_path, genres):
    """
    Generate and save spectrograms from 3-second audio clips without unnecessary visual elements.
    """
    for genre in genres:
        clips_path = os.path.join(base_path, 'dataset/audio3sec', genre)
        for filename in os.listdir(clips_path):
            song_path = os.path.join(clips_path, filename)
            y, sr = librosa.load(song_path)
            mels = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(mels, ref=np.max)
            
            fig, ax = plt.subplots()
            canvas = FigureCanvas(fig)
            ax.axis('off')  # Do not display axis
            librosa.display.specshow(S_dB, sr=sr, ax=ax)
            
            spectrogram_path = os.path.join(base_path, 'dataset/spectrograms/train', genre, filename.replace('.wav', '.png'))
            plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
            plt.close()
def distribute_train_test(base_path, genres, test_ratio=0.1):
    """
    Distribute spectrograms into training and test sets based on a specified ratio.
    """
    for genre in genres:
        train_path = os.path.join(base_path, 'dataset/spectrograms/train', genre)
        test_path = os.path.join(base_path, 'dataset/spectrograms/test', genre)     #TODO: improve directory structure 
        filenames = os.listdir(train_path)
        random.shuffle(filenames)
        test_size = int(len(filenames) * test_ratio)
        test_files = filenames[:test_size]
        for filename in test_files:
            shutil.move(os.path.join(train_path, filename), os.path.join(test_path, filename))

base_path = 'audiio_data_30_sec'  # Long audio directory
genres = 'blues classical country disco pop hiphop metal reggae rock'.split()   # subfolder names

create_directories(base_path, genres)
split_save_audio_files(base_path, genres)
generate_spectrograms(base_path, genres)
distribute_train_test(base_path, genres)
