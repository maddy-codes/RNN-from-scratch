from utils.utils import TextDataset
from torch.utils.data import DataLoader
from RNN.rnn import RNN, train
from torch import nn, optim
if __name__ == "__main__":
    with open('data/dinos.txt','r') as f:
        data = f.read()

    data.lower()

    seq_length = 25
    batch_size = 64
    hidden_size = 256

    text_dataset = TextDataset(data, seq_length=25)
    text_dataloader = DataLoader(text_dataset,batch_size=batch_size)

    #Model
    rnn_model = RNN(input_size=1, hidden_size=hidden_size, output_size=len(text_dataset.chars), batch_size=batch_size)

    #Train Variables 
    epochs = 1000
    loss = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(rnn_model.parameters(),lr=0.001)

    train(rnn_model,text_dataloader,epochs,optimizer,loss)