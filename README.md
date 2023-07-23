# MNet: V1.1 (An updated Framework to Reduce Fruit Image Misclassification) 


This repositoy cosnists of 1 python file: 
1) MNetV1.py

## Pranayama positions classification using modified MNet framework:
We have enhanced the classification of Pranayama positions as "Right" or "Wrong" by adapting the existing "MNet" framework, which can be found at: https://codeocean.com/capsule/1673587/tree/v1. Our modifications involve the integration of two distinct lightweight and highly precise pre-trained models: "MobileNet" and "NASNetMobile" (Reference: https://keras.io/api/applications/). Through these enhancements, the modified framework now exhibits increased accuracy and effectively mitigates the misclassification issue.

## Dataset 
Sample images are used from the original dataset in this project. the original dataset is
publicaly avaialbel on Mendley (https://data.mendeley.com/datasets/p2dhvhcw27/2) as a "Pranayama Dataset: A Collection of Breathing Exercise Visuals (Images and Videos) for Health and Wellness". The dataset is  The article realted to dataset is available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4473586.

# Code Files (.py programs)
### 1. MNetV1.py
The classification of Pranayama positions as "Right" or "Wrong" is accomplished by adapting the preexisting "MNet" framework. Unlike the original "MNet" that utilized a single pretrained model (InceptionV3) for fruit classification and quality assessment, our approach incorporates two separate pretrained models. One model is employed to identify the type of Pranayama, while the other determines whether the position is "Right" or "Wrong." These two models, namely "MobileNet" and "NASNetMobile," are not only lightweight but also boast remarkable accuracy levels. This dual-model strategy allows for a more precise and effective classification process.


### Hyperparameters tuning
Hyperparameters in Machine learning are those parameters that are explicitly defined by the user to control the learning process. some examles are: 1) Learning rate for training a neural network 2) Train-test split ratio 3) Batch Size 4) Number of Epochs. As per the rquirement user can chage these parameters in the code.
