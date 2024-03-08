# Music Genre Classification using CNN

Music plays a vital role in our lives, eliciting a wide range of emotions from nostalgia to excitement. This project focuses on classifying music genres using a Convolutional Neural Network (CNN), leveraging the unique properties of audio signals and transforming them into visual representations for effective genre classification.

## Project Overview

The goal of this project is to accurately classify music tracks into their respective genres by analyzing short clips of audio. We use a CNN model for this task, as CNNs are highly effective in extracting patterns from visual data. The process involves converting audio files into mel spectrogram images, which capture the essential frequencies and textures of the music, serving as the input data for our CNN model.

## Audio Dataset Info

### Download Link

[Download the dataset here](#)  <!-- Add your actual download link here -->

### Contents

The dataset is organized into 11 subfolders, each representing a unique music genre. Each subfolder contains 100 audio files, with each file having a duration of 30 seconds. The total dataset comprises 1100 audio files, providing a diverse collection of music samples for training and testing our model.

## Dataset Preprocessing

To prepare the dataset for training our CNN model, we perform several preprocessing steps:

1. **Audio Splitting**: Each 30-second audio file is split into 10 clips of 3 seconds each, resulting in 1000 audio samples per genre. This increases the number of training samples and allows the model to learn from a broader range of music snippets.

2. **Mel Spectrogram Conversion**: For each audio clip, we generate a mel spectrogram image. Mel spectrograms provide a visual representation of the audio's frequency content over time, which is crucial for genre classification.

3. **Train-Test Split**: The spectrogram images are shuffled and divided into training and testing sets. This split is essential for evaluating the model's performance on unseen data.

### Running the Preprocessing Script

To start the preprocessing, run the `audio_2_image_dataset_generator.py` script:

```bash
python audio_2_image_dataset_generator.py
```

## Image Dataset Info
After proessing complete, go to your audio data directory. Inside that you will find a new dataset folder. It contains the audio3sec folder and spectogram folder. Inside spectorgram, there is train and test split. Cut thso and put it in to root directory. 

You can download the processed dataset directly from here and start the training immediately. 
[Download the dataset here](https://drive.google.com/file/d/1_Yc3AGMdtZK9dAaIf2ufPy_pfL4Y1YLh/view?usp=sharing )  

## Train 
Run the following, it will train the CNN model and saved the model in the directory. 
```bash
python train.py
```

## Inferencing  
