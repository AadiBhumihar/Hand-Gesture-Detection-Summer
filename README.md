# Hand-Gesture-Detection-Summer Research Internship

Introduction :
Hand gesture detection can include combining hand shapes,orientation of hands to fluidly express speaker’s thought.So it is basically 
language interpretor for person with disabilities. This project focuses on gesture recognition and it uses computer vision and machine learning techniques 
to achieve this goal. We used three mathematical algorithms to train the parameter for different gesture.
there are three step to accomplish hand gesture detection system.
1-Hand detection
2-Feature extraction
3-Recognition

(1:Basic CNN Approach)
This is a readme file for Assignment Tree seadling classification

In our case we will use a very  convnet with  layers and  filters per layer, alongside data augmentation and dropout. 
Dropout helps to reduce overfitting, by preventing a layer from seeing twice the exact same pattern, thus acting in
a way analoguous to data augmentation

1:Layer contain 32 filter of (3*3) size with stride of 1 and relu activation function which fire those neurons which have negative value because 
relu = max(num,0)and then max pooling of (2,2) for translation and rotation invariant and reduce feature

2:Layer contain 64 filter of (3*3) size with stride of 1 and relu activation function which fire those neurons which have negative value because 
relu = max(num,0)and then max pooling of (2,2) for translation and rotation invariant and reduce feature

2:Layer contain 32 filter of (3*3) size with stride of 1 and relu activation function which fire those neurons which have negative value because 
relu = max(num,0)and then max pooling of (2,2) for translation and rotation invariant and reduce feature

then we connect it with fully connected neural network 

To do this first we reshape the parameter and made it flat 

Then Add hidden layer of 64 node and then dropout of 0.5 to reduce overfitting .

And final layer is of 6 node because of 6 class and activation function of 'Softmax Fuction ' and loss function of ='categorical_crossentropy' and 'rmsprop' optimizer .
Then Fit Our Model using mini-batch Sochastic Gradient Descent using rmsprop optimizer.
Here we use the data Augmentation Concept to increase our data by zooming,shearing,rescaling and fliping.

After this we train our data on gpu i get good result of above 85%;

(2: Using BottleNeck Feature)

In this we use Pre-trained VGC16 model trained on ImageNet dataset  to extract best feature from or data set and prevent to use Image Augementation .

After this we feed this feature into neural network with the  configuration :
First we reshape the parameter and made it flat 
Then Add hidden layer of 256 node and then dropout of 0.5 to reduce overfitting .
And final layer is of 12 node because of 12 class and activation function of 'Softmax Fuction ' and loss function of ='categorical_crossentropy' and 'rmsprop' optimizer .
Then Fit Our Model using mini-batch Sochastic Gradient Descent using rmsprop optimizer.
And to save model weight into a file.

(3: Using Fine-Tuning Model) :
To further improve our previous result, we can try to "fine-tune".In this model we dont randomly intilize weight but assign the weight that we obtained
from bottleneck model and Connect our pre-trained VGC16 model to neural network model with the  configuration :
First we reshape the parameter and made it flat 
Then Add hidden layer of 256 node and then dropout of 0.5 to reduce overfitting .
And final layer is of 12 node because of 12 class and activation function of 'Softmax Fuction ' and loss function of ='categorical_crossentropy' and 'rmsprop' optimizer .

and in this we dont train only first 25 layers of our model and after this we re-train all the layer of VGC16 and our neural network model.
the last convolutional block of the VGG16 model alongside the top-level classifier.
Then Fit Our Model using mini-batch Sochastic Gradient Descent using rmsprop optimizer.
But Last two method require High Memory Resource Requirment and need gpu to run.
