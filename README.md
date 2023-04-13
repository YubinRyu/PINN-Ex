# PINN-Ex
PINN-Ex is based on [physics-informed neural networks (PINN)](https://github.com/maziarraissi/PINNs) proposed by M.Raissi. 
This model predicts fluid flow and the reaction of a polymerization reactor depending on the operating conditions.
The adaptive sampling technique is also utilized to enhance prediction accuracy and convergence. 
The overall methodology employed in this study is demonstrated to be effective in predicting multi-physics results within the interpolation and the extrapolation range. 

## Installation
Run the following command to set up.

    git clone https://github.com/YubinRyu/PINN-Ex.git
    
You need to download preprocessed data or raw data to train this model. 

    python data/data_setup.py # Preprocessed data
    python preprocessing/raw_data/data_setup.py # Raw data

## Workflow
### 1) Data preprocessing
__If you downloaded the preprocessed data, you can skip data preprocessing. Since the file is too large, we recommend you to download the preprocessed data.__

Run the following command to preprocess the raw data. 

    python preprocessing/inner.py
    python preprocessing/outer.py
    python preprocessing/total.py

### 2) Training
__Since this model is based on distributed-data parallel (DDP), we recommend you to use 2 or more GPUs to train this model.__

Run the following command to train PINN-Ex. You can change hyperparameters (batch size, learning rate, epoch, activation function, etc.) by using Argument Parser. 

    python model/PINN/main_ddp.py
    
### 3) Results
#### 1. Best model
The model with the smallest validation loss was saved as the best model. 
- result/PINN/best_model.pth

#### 2. Checkpoint
The loss and validation loss for every epoch were saved to a checkpoint model. 
- result/PINN/checkpoint.pt

#### 3. Adaptive sampling points
When adaptive sampling was performed, an additional train dataset was saved in `X_{epoch}.csv`, and the corresponding target dataset was saved in `Y_{epoch}.csv`. 
- result/PINN/adaptive_sampling/points/X_{epoch}.csv
- result/PINN/adaptive_sampling/points/Y_{epoch}.csv

#### 4. Contour plot
Every 1000 epochs, the adaptively sampled data points and the velocity contour plot were saved as PNG files.
- result/PINN/adaptive_sampling/contour_plot/{epoch}.png
