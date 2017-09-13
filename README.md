
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/N001_model.png "Model Visualization"
[image2]: ./examples/center_2017_09_03_23_43_30_236.jpg "Center"
[image3]: ./examples/left_2017_09_03_23_43_30_236.jpg "Recovery Image Left"
[image4]: ./examples/right_2017_09_03_23_43_30_236.jpg "Recovery Image Right"
[image5]: ./examples/Figure_1-1.png "full"
[image6]: ./examples/Figure_1_augmentation2.png "Augmentation"

[image7]: ./examples/figure_1_model002.png
[image8]: ./examples/figure_1_model002_2.png
[image9]: ./examples/figure_1_model002_3.png
[image10]: ./examples/figure_1_model002_crop55-20.png
[image11]: ./examples/figure_1_model002_crop65-30.png
[image12]: ./examples/figure_1_model002_reconstructed.png
[image13]: ./examples/figure_1_model002_reconstructed2.png
[image14]: ./examples/figure_1_N001_3x6411_pp.png
[image15]: ./examples/figure_1_N001_3x6411_pp2.png
[image16]: ./examples/figure_1_N001_3x6411_pp_mb3.png
[image17]: ./examples/figure_1_N002_3x6411_pp.png
[image18]: ./examples/figure_1_N002_3x6411_pp_bilat.png
[image19]: ./examples/figure_1_N002_3x6411_pp_bilat5-75-25.png
[image20]: ./examples/figure_1_N002_3x6411_pp_bilat5-75-50.png
[image21]: ./examples/figure_1_N002_3x6411_pp_bilat5-100-25.png
[image22]: ./examples/figure_1_N002_pp.png
[image23]: ./examples/figure_1_N3_pp.png
[image24]: ./examples/figure_1_N003_pp.png
[image25]: ./examples/Figure_1.png
[image25]: ./examples/Figure_1-1.png

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video in basic quality [run1.mp4](./examples/run1.mp4) (default FPS=60)
* video in better quality [run2.mp4](./examples/run2.mp4) (default FPS=60)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 128 (clone.py lines 147-154)

The model includes RELU layers to introduce nonlinearity (code line 147-154), and the data is normalized in the model using a Keras lambda layer (code line 143).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 155).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 81-108). The data sets where created during execution time using a pre-processing pipeline of image augmentation filters (blur, median, bilateral, clahe). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I decided to implement a steering angle correction (angle -/+0.28) utilizing the left and right camera image (code line 60-65)

#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a well-known existing model, [NVIDIA E2E Network Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and iteratively change the structure and parameters minimizing the loss MSE. I thought this model might be appropriate because it has been developed for self-driving cars in real world, thus being able to handle even more complex scenario's as modeled in the simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set 70%/30% (code line 27). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I removed a big fully connected layer, added cropping, normalization and Dropout function (code line 154) helped too. Further early termination of epochs.

Then I modified the hyper-parameters (crops, dropout, correction, several parameter in the augmentation) to minimize the loss function.

The final step was to run the simulator to see how well the car was driving around track one. There were 2 spots where the vehicle fell off the track (sharp left turn after brigde, thereafter right turn at the lake) to improve the driving behavior in these cases, I added Conv-layers to deepen the model and squeeze the spacial dimensions. Further I needed to modify dimensions of the fully connected layer.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 6. Final Model Architecture

The final model architecture (clone.py code line 147-154) consisted of a convolution neural network with the following layers and layer sizes ...
____________________________________________________________________________________________________
###### Layer (type)         |  Output Shape      |    Param #   |  Connected to        |
====================================================================================================
cropping2d_1 (Cropping2D)      |  (None, 65, 320, 3) |   0    |       cropping2d_input_1     |
____________________________________________________________________________________________________
lambda_1 (Lambda)              |  (None, 65, 320, 3)  |  0        |   cropping2d_1       |
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D) | (None, 32, 159, 24) |  672     |    lambda_1           |
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D) | (None, 15, 79, 48)  |  10416   |    convolution2d_1     |
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D) | (None, 7, 39, 64)   |  27712   |    convolution2d_2     |
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D) | (None, 5, 37, 72)  |   41544   |    convolution2d_3      |
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)|  (None, 3, 35, 96)   |  62304   |    convolution2d_4      |
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  |(None, 1, 33, 96)   |  83040   |    convolution2d_5      |
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D) | (None, 1, 33, 128)  |  12416    |   convolution2d_6     |
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D) | (None, 1, 33, 128)  |  16512    |   convolution2d_7     |
____________________________________________________________________________________________________
dropout_1 (Dropout)        |      (None, 1, 33, 128)  |  0        |   convolution2d_8     |
____________________________________________________________________________________________________
flatten_1 (Flatten)        |      (None, 4224)      |    0       |    dropout_1          |
____________________________________________________________________________________________________
dense_1 (Dense)           |       (None, 200)       |    845000    |  flatten_1         |
____________________________________________________________________________________________________
dense_2 (Dense)          |        (None, 25)       |     5025     |   dense_1           |
____________________________________________________________________________________________________
dense_3 (Dense)          |        (None, 9)        |     234      |   dense_2           |
___________________________________________________________________________________________
 dense_4 (Dense)          |        (None, 1)        |     10     |     dense_3            |
_____________________________________________________________________________________
Total params: 1,104,885

Trainable params: 1,104,885

Non-trainable params: 0



Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 7. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then used the vehicle recovering effect modifying the steering angle for left and right camera images. These images where used.

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images (no. 1) and angles thinking that this would help for curves and would double the data prevent overfitting.

![alt text][image6]

no. 0 = original,
no. 1 = flipped,
no. 2 = blur,
no. 3 = median blur,
no. 4 = bilateral filter,
no. 5 = clahe filter

After the collection process, I had over 135600 of data points. I preprocessed the data during batches, ensuring test and validation would use same pipeline.

Further random examples

![alt text][image5]

I finally randomly shuffled the data set and put 30% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 (with sample_per_epoch:  5625) as evidenced by the graph

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]

![alt text][image19]

![alt text][image20]

![alt text][image21]

![alt text][image22]

![alt text][image23]

![alt text][image24]

