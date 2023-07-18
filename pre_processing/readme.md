We extract the speech and visual features automatically from videos using existing tools.

1. Download your videos. You will need videos in .mp4 format, annotations (IPU) in csv format and audio in .wav format.
2. Create a directory './data/[your_dataset]' and inside it the following structure :
```
data
-----[your_dataset]
--------rax_data
---------------Audio
---------------------processed
---------------------Annotations
---------------------WAV_16000
---------------Video
---------------------processed
---------------------Full
```
3. Download and install OpenFace using : https://github.com/TadasBaltrusaitis/OpenFace.git (the installation depends on your system).
4. Execute 'conda env create -f openface_env.yml'.
5. In the file "pre-processing/extract_openface.py" change "openFace_dir = "PATH/TO/OPENFACE/PROJECT" and put the absolute path of the project.
6. In the conda console, extract the speech features by executing "python pre_processing/extract_opensmile.py False".
7. In the conda console, extract the visual features by executing "python pre_processing/extract_openface.py False".
8. In the conda console, align the speech and visual modalities by executing "python pre_processing/create_set.py -dataset [your_dataset] -zeroMove
-zeroMove is used only if you want to set the behaviors to 0 when the person is not speaking. 
