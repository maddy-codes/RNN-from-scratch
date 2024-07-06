#Imports
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import cuda
from logging import Logger
class RNN(nn.Module):
    """
    RNN Block......... This represents a single layer of RNN.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size:int, batch_size:int)-> None:
        """
        This functions initialises the RNN block with a simple cell connection of input, hidden and output layer.

        Args:
        input_size: Number of features of your input vector
        hidden_size: Number of hidden neurons
        output_size: Number of output neuroms

        Return:
        Nothing
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False) #Initialisng the input layer to hidden layer connection.
        self.h2h = nn.Linear(hidden_size, hidden_size) #Initiaising the hidden to hidden layer connection.
        self.h2o = nn.Linear(hidden_size, output_size) #Initialising the hidden to output layer connection.

    def forward(self, x, hidden_state) -> tuple[torch.Tensor,torch.Tensor]:
        """
        This functions represents the forward pass in a RNN.

        Args:
        x: Input vector
        hidden_state: Previous State

        Return:
        out: Linear output (This is without activation)
        hidden_state: the new hidden state after going through the tanh activation function.
        """
        x = self.i2h(x) #initialising the inputs
        hidden_state = self.h2h(hidden_state) #initialising the hidden layer
        hidden_state = torch.tanh(x + hidden_state) #calculating the hidden state on the sum of input and the hidden_state
        out = self.h2o #initialising the output
        return out, hidden_state
    
    def init_zero_hidden(self,batch_size=1) -> torch.Tensor:
        """
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False) #simply initialising and returning zeros
    
def train(model: RNN, data: DataLoader, epochs:int, optimizer:optim.Optimizer,loss_fn: nn.Module)->None:
    """
    Trains the model for specified number of epochs.

    Args:
    model: initalised RNN Model
    data: data that will be fed into the model
    epochs: total number of epochs
    optimizer: the optimizer to be used for each epochs
    loss_fn: Function to calculate loss
    """
    train_losses = {} #this dictionary keeps track of the training loss at each epoch
    device = "cuda" if cuda.is_available() else "cpu" #it will pickup gpu if available otherwise it will pickup cpu.
    model.to(device)
    model.train() #traing the model.........
    print("#######Starting Training Procedure#######")
    for epoch in range(epochs):
        epoch_losses = list() #initalising the list for storinf each epoch losses
        for X,Y in data: #getting the training vs labels in the labels
            #skip last batch if it dosent match with the batch size 
            if X.shape[0] != model.batch_size:
                continue
            
            hidden = model.init_zero_hidden(batch_size=model.batch_size)
            #send tensors to device
            X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)
            #clear gradients
            model.zero_grad()

            loss = 0 #initisling the loss
            for c in range(X.shape[1]):
                out, hidden = model(X[:,c].reshape(X.shape[0],1),hidden)
                l = loss_fn(out, Y[:,c].long())
                loss += l

            #Compute the gradients
            loss.backward()

            #adjusting the learnable paramerters
            #clipping to avoid vanishing and exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(),3)
            optimizer.step()

            epoch_losses.append(loss.detach().item()/X.shape[1])
        train_losses[epochs] = torch.tensor(epoch_losses).mean()

        print(msg=f"epoch: {epoch + 1}, loss: {train_losses}")
        #after each epoch generate text
        #Logger.info(generate_text(model,data.dataset))