# Session-4 Assignment
## Problem Statement
You are making 3 versions of your 4th assignment's best model (or pick one from best assignments):
- Network with Group Normalization
- Network with Layer Normalization
- Network with L1 + BN

You MUST:
- Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include
- Write a single notebook file to run all the 3 models above for 20 epochs each
- Create these graphs:
  - Graph 1: Test/Validation Loss for all 3 models together
  - Graph 2: Test/Validation Accuracy for 3 models together 
  - Graphs must have proper annotation
- Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images. 

## Solution
### Part 1 - Single model.py file
A **model.py** file was created with a class Net (defining Model Architecture from last assignment) with following changes:
- A new function ***normalization*** was created with input variables
  - nm --> normalization technique to be used ('BN', 'LN', 'GN') (required)
  - group --> This works when Group Normalization is used (not required, Default value: 2)
- ***init*** function was same as best model in Assignment 4 with following changes:
  - Takes 2 more inputs:
    - nm --> normalization technique to be used ('BN', 'LN', 'GN') (not required: default value 'BN')
    - gp --> This works when Group Normalization is used (not required, Default value: 2)
- ***forward*** function has no changes
### Part 2 - Creating ipynb file on colab which uses model.py
- Model was imported in the colab using the following code:
```
from google.colab import drive
drive.mount('/content/drive')        #importing google drive to access model.py file
import sys
sys.path.append('/content/drive/MyDrive/EVA8/Session_5')     #path to the folder containing model.py file
import model          #importing model.py file
from model import Net          #importing model class defined as Net
```
- Function was written to diplay wrong predictions given 2 variables (wrong predictions and number of image to display (default 10)
- L1 loss was caluclated using similar formula mentioned in study material
```
loss = F.nll_loss(y_pred, target)    

l1 = 0
if lambda_l1 > 0:
    for p in model.parameters():
        l1 = l1 + p.abs().sum()

loss = loss + lambda_l1 * l1
```
- Model was trained 3 times for Batch Normalization with L1, Group Normalization (set at 2 groups), Layer Normalization
### Batch Normalization with L1
- Best Train Accuracy: 99.35%
- Best Test Accuracy: 99.41%
- Misclassified 10 images:
<img width="330" alt="BN_with_L1" src="https://user-images.githubusercontent.com/118976187/215042960-bc071cdf-4d0c-452f-88ca-faf0efe3f897.png">

### Group Normalization (set at 2 groups)
- Best Train Accuracy: 99.29%
- Best Test Accuracy: 99.35%
- Misclassified 10 images:
<img width="318" alt="GN" src="https://user-images.githubusercontent.com/118976187/215043003-076be3fe-fc78-4664-9a2c-2dc97cf863cd.png">

### Layer Normalization
- Best Train Accuracy: 99.23%
- Best Test Accuracy: 99.29%
- Misclassified 10 images:
<img width="323" alt="LN" src="https://user-images.githubusercontent.com/118976187/215043428-6ab78c85-1c98-4222-af9a-bf0e48601bfb.png">

### Findings
- Batch Normalization with L1 performed best in terms of Train as well as Test Accuracy
- This was expected as for CV problems the context of all images matters less than for NLP task, so generally it should perform the best
- Other Normalizations techniques were not far behind the reason for the same might be:
  - Relatively easy training and test set (with 0-9 digits hand written)
  - In complex scenarios of classifying images for multiple clothing types etc LN and GN might not work so well
- Training loss for BN+L1 is quite high compared to the other two normalizations. Training loss follows a nearly similar pattern for LN & GN.
- Test loss is maximum for LN, followed by GN & lowest for BN+L1.
- Training Accuracy curve is nearly the same for all of three normalizations.
- Test Accuracy is highest for BN+L1, follwed by GN & lowest for LN

### Graphs (Loss and Accuracies)
<img width="903" alt="Graphs" src="https://user-images.githubusercontent.com/118976187/215046490-a1c82592-0102-4e4c-8efa-6d97bbaff7f2.png">

## Normalization Techniques (Batch Normalization, Layer Normalization, Group Normalization)
### Learnings of Normalization Techniques
- Normalization is used to normalize weights to keep them in smaller scale
- This helps in weights not be amplified. If a weight in some earlier layer is very high in some channel then it will cause amplification in all the layers.
- Normalization drives this problem down by normalizing the weights
- Batch normalization works channel wise, i.e, it looks at one channel at a time for entire batch
- Layer normalization works across one train data across all channels
- Group normalization also works single train data but the channels can be divided into multiple groups
- The purpose of usages can be different for all 3 normalizations
- The Computer Vision tasks generally use Batch Normalization while Layer and Group Normalizations are used in NLP (Natural Language Processing) as context is more important in NLP tasks

### Examples
- Batch Normalization
<img width="1342" alt="Batch Norm" src="https://user-images.githubusercontent.com/118976187/215050800-0e121fe5-78e0-48e5-8682-fa448a811d71.png">

- Layer Normalization
<img width="1322" alt="Layer Norm" src="https://user-images.githubusercontent.com/118976187/215050829-311544b8-0b31-454a-b6b9-e9b716e314a3.png">

- Group Normalization
<img width="1337" alt="Group Norm" src="https://user-images.githubusercontent.com/118976187/215050847-b9969df7-29e4-4bb7-a46c-645f5106e359.png">
