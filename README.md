# Graph Neural Network

Some part of the code kept private because paper is not published yet.

**GNN model**

GNN model consists of graph transformer layer then linear transformation applied on the output and at the end batch normalisation. I have stacked these layers several times and applied TopK pooling to reduce the size of the graph. The intermediate embedding after each pooling step is aggregated into a global graph representation.

The final graph representation is fed to a couple of linear transformations, using the ReLU activation function.

**Graph features**

The MoleculeDataset class will generate the features of the molecules from the SMILES using the deepchem package.

The idea is to generate the features and save (cached) them into a file so that repeated generation of features can be avoided. We can skip this step by directly generating the features and using them for the model building, but if the RAM size is small, it can cause breaking of training or slowing it.

**Model training**

We are training the model for 50 Epoch. The train_loader will load the data into batches and train the model for each epoch. Optimizer is set to zero_grad to avoid the accumulation of the previous loss.

In forward Pass the features are fed to the input layer in the model. The model calculates the respective weighted sum and passes it to the activation function to add non-linearity. In the end, the loss is calculated by comparing the measured value with the predicted.

Once the loss is calculated it is backpropagated using the backward() function to adjust the respective weights. backpropagation aims to minimize the cost function by adjusting the network’s weights and biases. The level of adjustment is determined by the gradients of the cost function with respect to those parameters.


**Run model**

Repository can be downloaded and run using following command.

python train.py
