# Names Classification using character-level LSTM model

<img src="https://github.com/Arushi04/character-level-RNN/blob/master/images/cover.png" width="800" height="400">

### Description : 
A character-level LSTM reads words as a series of characters and outputs a prediction and a “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to. In this project, we will train a few thousand names from 18 languages of origin, and predict which language a name is from based on the spelling.

### Dataset:
The dataset consists of names from 18 languages of origin and can be found in the data folder. There are 18 files of different languages, consisting of one name in each line.

<img src="https://github.com/Arushi04/character-level-RNN/blob/master/images/lstm.jpg" width="700" height="300">

### Relevant Files:

The project is broken down in 3 files:

**dataset.py :** Loading, pre-processing and splitting the dataset using Data Loader
**model.py :** Defining the layers of the LSTM model
**main.py :** Training the training dataset using the defined model and predicting classes for test images. Visualizing traing and test loss and accuracy on test datasets

### Requirements
* Python 3.6.10  
* Numpy 1.18.4  
* Tensorboard 2.0.0   
* Pytorch 1.5.0  
* Torchvision 0.6.0 
* Matplotlib 3.2.1
* Scikit-learn 0.23.1   

### Command to Run:

python main.py \      
--datapath data/names \    
--outdir output/ \    
--epochlen 13 \    
--modelname modelv \     
--lr 0.05 \     
--embed_dim 50 \     
--hidden_size 100    










