# Hand-Written-To-Text-Recognition

Links to the data sets are

[Data set Link for SVM-KNN](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset)

Small Letters [Dataset](https://ai.stanford.edu/~btaskar/ocr/) for CNN

Capital Letters [Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition) for CNN

#### Image pre-processing

Download the second datasets to the current working directory, Run the code `data_maker.ipynb` to generate `char-test.csv` and `char-train.csv` files, which are currently available in the repository.

#### Models Training and saving

Now we can train the models by running `CNN_5_fold.ipynb` By doing this the models will be saved locally in the folder `models` which is now currently in the repository renamed as `models_without_numbers`. So for running the code `CNN_5_fold.ipynb` you have to rename it as  `models`.

By running this code You will get a pop-up of the confusion matrix. In this directory named as `CM.png`.

#### Image to Text prediction

To do this you have to run the file `predictor.ipynb` This asks for path of the input image and then starts segmenting image. After segmenting the image we will get an image `segmented_image.png` and for each of the you'll get images `segment_no_{num}.png`.

And the predicted text will be printed finally.

Or just run the command

```
$python3 predictor.py
```

The output for the `test-case.png` is `ViVek HrisWitha Siddhj SoHITH` Which is not 100% accurate but is as close as we get near it.

The folder `Gen_Sentense` contains file `Gen_img.py` by running this, we can generate the image `test-case.png` which can be used to test the predictor.


#### Other Dataset and SVM

You can download the First dataset and run the code `image_to_csv.ipynb` to get `test.csv` and `train.csv` files. Those csv files can used to run the KNN-SVM python file. 

You can check the accuracy of this model too.

The same files also can be used to train the CNN files but.

Here is the link to the detailed report for our project.[Project Report](/home/vivek/Documents/AI ML/Project/modified/img_to_csv.ipynb)
