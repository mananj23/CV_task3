# CV_task3
- Computer Vision - task 3 for pclub secy recruitment - MANAN JINDAL 230625
- I did coding in google colab.
# Finding data set
 - I found a data set on Kaggle('https://www.kaggle.com/datasets/yasserhessein/gender-dataset') 
 - Data description :
    - The data collection contains cropped pictures of male and female subjects.
    - It is divided into training and validation directories.
    - The training directory provides 23,000 photos of each class, whereas the validation directory has 5,500 images of each class.
  - This data set didn't include masked people so i had to look up for another dataset which i also found on Kaggle ("https://www.kaggle.com/datasets/itsshuvra/gender-classified-dataset-with-masked-face").
  - Data description :
    - There are 89,443 pictures in total for training, which are divided into two groups (male and female). Additionally, there are 20,714 pictures for validation.
  - These datasets had alot of data so i had to make my own dataset that contains a mixture of both the data sets in sufficient quantities.
  - Google Drive link - ("https://drive.google.com/file/d/1SqXnVBAY0RpIuvBie9hvdgfQh4wryp5b/view?usp=sharing")
# Categorising data
 - I categorised my dataset by reading file_name present in inside the folder by importing os.<br>
   {'Male': 0, 'Female': 1}
# Reading Face
 - For this, i used a pre-trained XML file that contains the configuration of a Haar cascade classifier specifically trained for detecting frontal faces in images.
 - The haarcascade_frontalface_default.xml file contains the parameters and structure of the trained classifier for detecting frontal faces.
 - So with the help of CascadeClassifier ,a function of cv2, i stored a gray resized 32X32 image in the data directory.
 - And the corresponding label is stored in a target directory
 - I also normalize my image data between 0 and 1 so i divided it by 255
 - I also had to keep in mind that my target is categorical.
# Applying CNN model
 - I used the Convolutional Neural Network (CNN) model using the Keras library.
 - I first fixed all of the hyperparameters that includes number of filters (or kernels) in the convolutional layers, the size of the convolutional filters in the first two 
   convolutional layers, size of the convolutional filters in the last two convolutional layers, size of the max-pooling window, and number of nodes in the fully connected 
   layers.
 - Then I created a sequential model object, using sequential().
 - Then i defined the Model Architecture using Convolutional Layers, Dropout Layers and Fully Connected Layers.
 - And finally i compiled my model.
 - I also used ModelCheckpoint, this callback function monitors the validation loss during training and saves the model weights to a file when the validation loss improves.
 - I also created an object called 'history' containing all the information about the training process, such as loss and accuracy metrics over epochs.
 - I got the following accuracies in epochs from 1 to 20.<br>

 Epoch 1/20<br>
 500/500 [==============================] - 54s 103ms/step - loss: 0.5233 - accuracy: 0.7200 - val_loss: 0.3161 - val_accuracy: 0.<br>
 Epoch 2/20<br>
 500/500 [==============================] - 51s 101ms/step - loss: 0.3167 - accuracy: 0.8633 - val_loss: 0.2704 - val_accuracy: 0.8801<br>
 Epoch 3/20<br>
 500/500 [==============================] - 49s 98ms/step - loss: 0.2683 - accuracy: 0.8835 - val_loss: 0.2241 - val_accuracy: 0.9009<br>
 Epoch 4/20<br>
 500/500 [==============================] - 53s 106ms/step - loss: 0.2350 - accuracy: 0.8991 - val_loss: 0.2178 - val_accuracy: 0.9011<br>
 Epoch 5/20<br>
 500/500 [==============================] - 51s 102ms/step - loss: 0.2187 - accuracy: 0.9070 - val_loss: 0.1999 - val_accuracy: 0.9124<br>
 Epoch 6/20<br>
 500/500 [==============================] - 48s 95ms/step - loss: 0.2044 - accuracy: 0.9138 - val_loss: 0.2045 - val_accuracy: 0.9054<br>
 Epoch 7/20<br>
 500/500 [==============================] - 49s 98ms/step - loss: 0.1862 - accuracy: 0.9204 - val_loss: 0.2010 - val_accuracy: 0.9136<br>
 Epoch 8/20<br>
 500/500 [==============================] - 51s 101ms/step - loss: 0.1785 - accuracy: 0.9236 - val_loss: 0.1842 - val_accuracy: 0.9206<br>
 Epoch 9/20<br>
 500/500 [==============================] - 50s 101ms/step - loss: 0.1687 - accuracy: 0.9287 - val_loss: 0.1802 - val_accuracy: 0.9226<br>
 Epoch 10/20<br>
 500/500 [==============================] - 49s 99ms/step - loss: 0.1588 - accuracy: 0.9327 - val_loss: 0.1749 - val_accuracy: 0.9276<br>
 Epoch 11/20<br>
 500/500 [==============================] - 48s 97ms/step - loss: 0.1530 - accuracy: 0.9339 - val_loss: 0.2073 - val_accuracy: 0.9109<br>
 Epoch 12/20<br>
 500/500 [==============================] - 49s 97ms/step - loss: 0.1433 - accuracy: 0.9405 - val_loss: 0.1706 - val_accuracy: 0.9306<br>
 Epoch 13/20<br>
 500/500 [==============================] - 51s 101ms/step - loss: 0.1383 - accuracy: 0.9407 - val_loss: 0.1810 - val_accuracy: 0.9261<br>
 Epoch 14/20<br>
 500/500 [==============================] - 49s 97ms/step - loss: 0.1320 - accuracy: 0.9437 - val_loss: 0.1699 - val_accuracy: 0.9266<br>
 Epoch 15/20<br>
 500/500 [==============================] - 50s 100ms/step - loss: 0.1227 - accuracy: 0.9490 - val_loss: 0.1743 - val_accuracy: 0.9314<br>
 Epoch 16/20<br>
 500/500 [==============================] - 48s 96ms/step - loss: 0.1160 - accuracy: 0.9502 - val_loss: 0.1738 - val_accuracy: 0.9311<br>
 Epoch 17/20<br>
 500/500 [==============================] - 49s 98ms/step - loss: 0.1234 - accuracy: 0.9502 - val_loss: 0.1987 - val_accuracy: 0.9249<br>
 Epoch 18/20<br>
 500/500 [==============================] - 48s 96ms/step - loss: 0.1119 - accuracy: 0.9524 - val_loss: 0.1830 - val_accuracy: 0.9316<br>
 Epoch 19/20<br>
 500/500 [==============================] - 54s 108ms/step - loss: 0.1055 - accuracy: 0.9539 - val_loss: 0.2006 - val_accuracy: 0.9301<br>
 Epoch 20/20<br>
 500/500 [==============================] - 48s 96ms/step - loss: 0.0975 - accuracy: 0.9601 - val_loss: 0.2021 - val_accuracy: 0.9276
 - Average training accuracy: 91.83320850133896 %
 - Average validation accuracy: 91.61116629838943 %
# Loading saved model
 - Then i loaded a saved Keras model from the file path ("./training/model-007.model"), that we pre-trained in the last step.
 - And also reloaded the haarcascade_frontalface_default.xml file.
 - Now our model is ready to use.
# Gender detection
 - Finally, i made the function gender_detection that takes input of images and does the following process:
   -  Read the image using open cv.
   -  Converts to gray scale.
   -  Detects faces using Cascade Classifier.
   -  Normalizes and reshapes the faces.
   -  Produces result using our pre-trained model.
   -  And finally draw rectangles and gender on the image.
 # Some results
  - There are two ways you can print output using the function:
     - Upload a zip file.(as I have done in ipnb file)
     - Take live photos and capture.(as shown in the example in the ipynb file)
       
   

