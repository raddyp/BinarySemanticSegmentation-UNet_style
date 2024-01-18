**Binary Semantic Segmentation-UNet style**  
U-Net style Binary Semantic Segmentation model for Forest Cover Dataset

This is a simple implementation of a U-net style segmentation architecture as per the "U-Net: Convolutional Networks for Biomedical Image Segmentation
" by Ronneberger et al.

This model was tested on the forest cover dataset as mentioned below.
Some parameters and architecture have been slightly modified to suit this dataset.

Dataset:  
A version of "DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images" was obtained from kaggle here:
https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation

Challenges:  
1. U-Net style architecture is computationally very expensive.
2. Image size and number of examples need to be chosen carefully to be able to train on a given machine.
3. For reliable training without crashing the OS with memory allocation errors and overrunning the RAM allocation, image size reduction is highly recommended.
4. Batch size is also a major factor in managing GPU memory loads during training (lower the better).

Code Description:
1. Function: display 
    To plot/display multiple images ina grid
2. Function: preprocess_datafrom_folder 
    To read, resize, and vectorize data
3. Function: read_data 
    To read image & mask data from 'h5' file from directory
4. Function: split_data 
    To split image & mask data into train, validation & test sets
5. Function: create_model 
    Defines the entire U-Net style model
6. Function: compute_metrics 
     To compute performance metric IoU
 7. Function: visualize_plots
    To visualize training accuracy & loss plots

Files:
1. Semantic_segmentation.py - implementation of the U-Net style segemntation model
2. acc.png - Training & validation accuracy plot
3. loss.png - Training & validation loss plot
4. model_sumamry.txt - Model summary
5. epochs.txt - Training epochs data

Results:
The model as tested in the script produced mixed resutls as it is not extensively trained and tested fo rthis dataset.
However there are some good results produced as show below. Teh shortcomigns in the results can be attributed to:
1. Architecture nto fully customized for this dataset
2. Images have relatively poor quality
3. Regions of interest are not substantially distinct form background in many images
4. 
Good segmentation predictions
![res4](https://github.com/raddyp/BinarySemanticSegmentation-UNet_style/assets/150963154/67ec7b56-506b-431d-be0d-8a8206905ace)
![res3](https://github.com/raddyp/BinarySemanticSegmentation-UNet_style/assets/150963154/57d846f6-832b-4d21-b793-5f351ae5f7d4)
![res1](https://github.com/raddyp/BinarySemanticSegmentation-UNet_style/assets/150963154/ed6a9994-98f9-425e-94a4-cad3a7c94139)

Bad segmentation predictions
![res5](https://github.com/raddyp/BinarySemanticSegmentation-UNet_style/assets/150963154/773d61c2-73e2-4cef-9b7a-a27cea21cb02)
![res6](https://github.com/raddyp/BinarySemanticSegmentation-UNet_style/assets/150963154/117e4457-096d-475a-8333-41a24a41e86d)
![res7](https://github.com/raddyp/BinarySemanticSegmentation-UNet_style/assets/150963154/c401d6cb-d904-4af3-9786-c6ede245dc04)






