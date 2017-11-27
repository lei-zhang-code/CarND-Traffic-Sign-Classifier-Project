#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./raw_data_visualization.png "Visualization"
[image2]: ./raw_data_histogram.png "Histogram"
[image3]: ./data_augmentation.png "Data augmentation"
[image5]: ./train_valid_accuracy.png "Training process"
[image6]: ./traffic_signs.png "Web downloaded traffic signs"
[image7]: ./prediction_probablity.png "Prediction probability"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lei-zhang-code/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

This figure shows one image for each class:

![alt_text][image1]

It is a bar chart showing the number of samples in each class. The number of samples are very unevenly distributed.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale contains sufficient information for human to recognize all the classes in the dataset. Removing redundant color information allows faster training speed.

As a last step, I normalized the image data to prevent gradient descent from overshooting and allow better convergence rate.

I decided to generate additional data because the model was overfitting. 

To add more data to the the data set, I used the following techniques: add Gaussian noise, scaled the image up and down, rotated the image by 5 degree counter-clockwise and clockwise.

The following figure shows the orignal RGB image of the wild-animal-crossing sign, grayscaled and normalized image, noise added image, scaled up image, and rotated image.

![alt_text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 12x12x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		|   400x172						|
| RELU					|												|
| Dropout               | 0.5 keep probablity  |
| Fully connected		|   172x86						|
| RELU					|												|
| Dropout               | 0.5 keep probablity  |
| Fully connected		|   86x43						|
| Softmax				|         									|
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, learning rate 0.001, epochs 6, batch size 128, and dropout keep probability 0.5. The following is the training process showing training set accuracy (cyan) and validation set accuracy (magenta) over time.

![alt_text][image5]

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.96875
* validation set accuracy of 0.96848
* test set accuracy of 0.94489

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet
* What were some problems with the initial architecture? 
Overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Added dropout layers at the two fully-connnected layers.
* Which parameters were tuned? How were they adjusted and why?
NA
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Do not add dropout layers to the convolution layers. I don't know why, but adding dropout layers to the convolution layers causes model to stop learning at ~90% accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt_text][image6]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Wild animal crossing  			| Wild animal crossing								|
| No passing					| No passing											|
| 30 km/h	      		| 30 km/h					 				|
| Bumpy road			| Bumpy road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.489%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all 5 images, the prediction probablities are all above 95%. The following is the figure showing the probabilities of top 5 maximum predictions of each image.

![alt_text][image7]

As a summary, the following the prediction of each image and their probability.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| Road work   									| 
| .999     				| Wild animal crossing 										|
| .999					| No passing											|
| 1.000	      			| 30 km/h					 				|
| .959				    | Bumpy road      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


