import torch
import numpy as np
import pandas as pd 
from tqdm import tqdm
from model import GNN
import torch_geometric
from features import MoleculeDataset
from torch_geometric.data import DataLoader


torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

params = {
    "batch_size": [128],
    "learning_rate": [0.01],
    "weight_decay": [0.0001],
    "sgd_momentum": [0.8],
    "scheduler_gamma": [0.8],
    "pos_weight": [1.3],
    "model_embedding_size": [64],
    "model_attention_heads": [3],
    "model_layers": [4],
    "model_dropout_rate": [0.2],
    "model_top_k_ratio": [0.5],
    "model_top_k_every_n": [1],
    "model_dense_neurons": [256]
}


def train(train_loader,train_dataset, num_epochs):

    num_examples = len(train_dataset)

    epochs = num_epochs
    num_batches = num_examples / 128
    losses = []

    for e in range(epochs):
        cumulative_loss = 0
        step = 0
        # inner loop
        for _, batch in enumerate(tqdm(train_loader)):
            batch.to(device)
        
            # Reset gradients
            optimizer.zero_grad() 
            pred = model(batch.x.float(),batch.edge_attr.float(),batch.edge_index,batch.batch)
        
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            loss.backward()
            optimizer.step()
        
            # Update tracking
            cumulative_loss += loss.item()
            step += 1
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
        losses.append(cumulative_loss / num_batches)

    return model

def test(model,test_loader):

    preds = []
    for batch in test_loader:    
        with torch.no_grad():
            batch.to(device)
            batch_preds = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)

    test_pred = pd.DataFrame(preds,columns=['Activity'])
    test_pred.to_csv('test_prediction.csv', index=False)

def main():
    # Load training data
    train_dataset = MoleculeDataset(root="data/", filename="train_set_data.csv")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    ## Define model parameters
    params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}

    ## Initialize GNN model
    model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params) 
    ## Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,  weight_decay=0.0001)
    ## Define loss function to optimize
    loss_fn = torch.nn.MSELoss()

    ### Train GNN transformer model
    model = train(train_loader,train_dataset, num_epochs=50)

    ## load test data for the prediction
    test_dataset = MoleculeDataset(root="data/", filename="test_set_data.csv", test=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    test(model,test_loader)
  
if __name__=='__main__':
    main()
