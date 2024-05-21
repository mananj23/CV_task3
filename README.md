# CV_task3
- Computer Vision - task 3 for pclub secy recruitment - MANAN JINDAL 230625
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
  - google drive link - ("https://drive.google.com/file/d/1SqXnVBAY0RpIuvBie9hvdgfQh4wryp5b/view?usp=sharing")
# Categorising data
 - I categorised my dataset by reading file_name present in inside the folder by importing os.<br>
   {'Male': 0, 'Female': 1}
# Reading Face
 - For this i used a pre-trained XML file that contains the configuration of a Haar cascade classifier specifically trained for detecting frontal faces in images.
 - The haarcascade_frontalface_default.xml file contains the parameters and structure of the trained classifier for detecting frontal faces.
 - So with the help of CascadeClassifier ,a function of cv2, i stored a gray resized 32X32 image in the data directory.
 - And the corresponding label is stored in a target directory
 - I also normalise my image data between 0 and 1 so i divided it by 255
 - I also had to keep in mind that my target is categorical.
# Applying CNN model
 - I used the Convolutional Neural Network (CNN) model using the Keras library.
 - I first fixed all of the hyperparameters that includes number of filters (or kernels) in the convolutional layers, the size of the convolutional filters in the first two 
   convolutional layers,size of the convolutional filters in the last two convolutional layers, size of the max-pooling window, and number of nodes in the fully connected 
   layers.
 - Then I created a sequential model object, using sequential().
 - Then i defined the Model Architecture using Convolutional Layers,Dropout Layers and Fully Connected Layers.
 - And finally i compiled my model.
 - I also used ModelCheckpoint, this callback function monitors the validation loss during training and saves the model weights to a file when the validation loss improves.
 - I also created an object called 'history' containing all the information about the training process, such as loss and accuracy metrics over epochs.
 - 
# Loading saved model
 - Then i loaded a saved Keras model from the file path ("./training/model-007.model"), that we pre-trained in the last step.
 - And also reloaded the haarcascade_frontalface_default.xml file.
 - Now our model is ready to use.
# Gender detection
 - 

