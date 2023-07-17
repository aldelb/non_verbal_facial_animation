# generation_of_facial_animation

The code contains a models to jointly and automatically generate the rythmic head, facial and gaze movements (non-verbal behaviours) of a virtual agent from acoustic speech features. The architecture is an Adversarial Encoder-Decoder. Head movements and gaze orientation are generated as 3D coordinates, while facial expressions are generated using action units based on the facial action coding system. 

## Example


https://github.com/behavioursGeneration/generation_of_facial_animation/assets/110098017/a8d5104d-7115-4d01-8736-f7659e2ec4ee

To see examples with natural voices, go [here](https://www.youtube.com/playlist?list=PLRyxHB7gYN-BPB6RvTt2xPE9nwLuMq2yD).

## The architecture 
![model](https://github.com/behavioursGeneration/generation_of_facial_animation/assets/110098017/69b9be4d-a048-47db-8d5d-f8996821c802)

## To reproduce
1. Clone the repository
2. In a conda console, execute 'conda env create -f environment.yml' to create the right conda environment. Go to the project location.

### Data and features extraction 
We extract the speech and visual features automatically from these videos using existing tools. You can also directly recover the extracted and align features with the next section.

1. Download a dataset, for example named "trueness"
2. Create a directory './data/trueness_data' and inside it, a directory "Features". Put the downloaded zip into "data", then unzip it. You must obtain the following structure :
```
data
-----trueness_data
--------raw_data
---------------Audio
--------------------Annotations
---------------Video
```
3. Download and install OpenFace using : https://github.com/TadasBaltrusaitis/OpenFace.git (the installation depends on your system). 
4. In the file "pre-processing/extract_openface.py" change "openFace_dir = "PATH/TO/OPENFACE/PROJECT" and put the absolute path of the project.
5. In the conda console, extract the speech features by executing "python pre_processing/extract_opensmile.py false".
6. In the conda console, extract the visual features by executing "python pre_processing/extract_openface.py false".
7. In the conda console, align the speech and visual modalities by executing "python pre_processing/align_data_mmsdk.py".
   

### Models training
1. In the directory "generation", you will find "params.cfg". 
It is the configuration file to customise the model before training. 
To learn what section needs to be change go see [the configuration file](docs/config_file.md).
2. You can conserve the existing file or create a new one. 
3. In the conda console, train the model by executing "python PATH/TO/PROJECT/train.py -params PATH/TO/CONFIG/FILE.cfg [-id NAME_OF_MODEL]"
You can visualise the created graphics during training in the repository "./generation/saved_models". 

### Behaviours generation
1. In the conda console, generate behaviours by executing "python PATH/TO/PROJECT/generation/generate.py -epoch [integer] -params PATH/TO/CONFIG/FILE.cfg -dataset [dataset]". The behaviours are generated in the form of 3D coordinates and intensity of facial action units. These are csv files stored in the directory "./generation/data/output/MODEL_PATH".

- -epoch : during training, if you trained in 1000 epochs, recording every 100 epochs, you must enter a number within [100;200;300;400;500;600;700;800;900;1000].
- -params : path to the config file. 
- -dataset : name of the considered dataset. 

### Models evaluation
The objective evaluation of these models is conducted with measures such as density evaluation, curves visualisation, visualisation from PCA reduction and jerk and acceleration measurements. 

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
