# generation_of_facial_animation
The code contains a models to jointly and automatically generate the rythmic head, facial and gaze movements (non-verbal behaviors) of a virtual agent from acoustic speech features. The architecture is an Adversarial Encoder-Decoder. Head movements and gaze orientation are generated as 3D coordinates, while facial expressions are generated using action units based on the facial action coding system. 

## Example

https://github.com/behavioursGeneration/generation_of_facial_animation/assets/110098017/a8d5104d-7115-4d01-8736-f7659e2ec4ee

The video below is generated in English with speech synthesized from text (TTS). Please note, that the model is trained exclusively with natural French voices. 
To see examples with natural voices, go [here](https://www.youtube.com/playlist?list=PLRyxHB7gYN-BPB6RvTt2xPE9nwLuMq2yD).

## The architecture 
![big_model](https://github.com/behavioursGeneration/generation_of_facial_animation/assets/110098017/8545d710-9e41-4235-ad46-515f57dc9301)

## To reproduce
1. Clone the repository
2. In a conda console, execute 'conda env create -f environment.yml' to create the right conda environment.
3. You will also need to execute 'conda env create -f openface_env.yml' if you want to extract yourself the behavioral features from videos.

### Features recovery 
You can also directly recover the extracted and align features with this section.

We extract the speech and visual features automatically from these videos using existing tools, namely OpenFace and OpenSmile. You can of course use the code in the [pre_processing](https://github.com/behavioursGeneration/generation_of_facial_animation/tree/main/pre_processing) folder to extract your own features from choosen videos. Please contact the authors to obtain the Trueness and/or Cheese datasets.

1. Create a directory './data'.
2. Download files found in [this drive](https://drive.google.com/drive/u/0/folders/16lF-p1wGfD3k9iVlrpTJ0A1mcZ92hvmj) for Trueness and in [this drive](https://drive.google.com/drive/u/0/folders/1XY9OMkyqPBBvl48zc7GigxaGRrkGEVC-) for Cheese and place them in the repository.

Files with the suffix "moveSpeakerOnly" are those whose behaviors are set to 0 if the person doesn't speak.

### Models training
1. "params.cfg" is the configuration file to customise the model before training. 
To learn what section needs to be change go see [the configuration file](docs/config_file.md).
2. You can conserve the existing file or create a new one. 
3. In the conda console, train the model by executing "python PATH/TO/PROJECT/generation/train.py -params PATH/TO/CONFIG/FILE.cfg [-id NAME_OF_MODEL]"
You can visualise the created graphics during training in the repository [saved_path] of your config file. By default "./generation/saved_models". 

### Behaviours generation
1. In the conda console, generate behaviours by executing "python PATH/TO/PROJECT/generation/generate.py -epoch [integer] -params PATH/TO/CONFIG/FILE.cfg -dataset [dataset]". The behaviours are generated in the form of 3D coordinates and intensity of facial action units. These are csv files stored in the the repository [output_path] of your config file. By default "./generation/data/output/MODEL_PATH".

- -epoch : during training, if you trained in 1000 epochs, recording every 100 epochs, you must enter a number within [100;200;300;400;500;600;700;800;900;1000].
- -params : path to the config file. 
- -dataset : name of the considered dataset. 

### Models evaluation
The objective evaluation of these models is conducted with measures such as dtw, curves visualisation, visualisation from PCA reduction and jerk and acceleration measurements. 

1.  In the conda console, evaluate model objectively by executing "python generation/evaluate.py -params PATH/TO/CONFIG/FILE.cfg -epoch [integer] -[PARAMETERS] "

- -params : path to the config file. 
PARAMETERS :
- -dtw'
- -pca
- -curve
- -curveVideo
- -acceleration
- -jerk

You will find the results in the directory "./generation/evaluation".


### Animate the generated behaviours
To animate a virtual agent with the generated behaviours, we use the GRETA platform. 

1. Download and install GRETA with "gpl-grimaldi-release.7z" at https://github.com/isir/greta/releases/tag/v1.0.1.
2. Open GRETA. Open the configuration "Greta - Record AU.xml" already present GRETA. 
3. Use the block "AU Parser File Reader" and "Parser Capture Controller AU" to create the video from the csv file generated.

4. ### Add voice 
You can directly concatenate the voices from the original videos to the Greta generated .avi videos.

```
input_video = "video_path.avi"
input_audio = "audio_path.wav"

output = "video_path_with_sound.mp4"

if(os.path.isfile(input_video) and os.path.isfile(input_audio)):
     audio = mp.AudioFileClip(input_audio)
     video = mp.VideoFileClip(input_video)
     final = video.set_audio(audio)

     final.write_videofile(output)
```
