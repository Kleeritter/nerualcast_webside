import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader


## Instead of coding an LSTM by hand, let's see what we can do with PyTorch's nn.LSTM()
class LightningLSTM(L.LightningModule):

    def __init__(
            self):  # __init__() is the class constructor function, and we use it to initialize the Weights and Biases.

        super().__init__()  # initialize an instance of the parent class, LightningModule.

        #seed_everything(seed=42)

        ## input_size = number of features (or variables) in the data. In our example
        ##              we only have a single feature (value)
        ## hidden_size = this determines the dimension of the output
        ##               in other words, if we set hidden_size=1, then we have 1 output node
        ##               if we set hiddeen_size=50, then we hve 50 output nodes (that can then be 50 input
        ##               nodes to a subsequent fully connected neural network.
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    def forward(self, input):
        ## transpose the input vector
        input_trans = input.view(len(input), 1)

        lstm_out, temp = self.lstm(input_trans)

        ## lstm_out has the short-term memories for all inputs. We make our prediction with the last one
        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):  # this configures the optimizer we want to use for backpropagation.
        return Adam(self.parameters(), lr=0.1)  ## we'll just go ahead and set the learning rate to 0.1

    def training_step(self, batch, batch_idx):  # take a step during gradient descent.
        input_i, label_i = batch  # collect input
        output_i = self.forward(input_i[0])  # run input through the neural network
        loss = (output_i - label_i) ** 2  ## loss = squared residual

        ###################
        ##
        ## Logging the loss and the predicted values so we can evaluate the training
        ##
        ###################
        self.log("train_loss", loss)

        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss


inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)
model = LightningLSTM()
## print out the name and value for each parameter
print("Before optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, param.data)

trainer = L.Trainer(max_epochs=4000, log_every_n_steps=2, accelerator="auto",devices="auto")
trainer.fit(model, train_dataloaders=dataloader)

print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
