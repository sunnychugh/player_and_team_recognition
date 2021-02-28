import pretrainedmodels
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, pretrained, model_name, teams_dic_len, players_dic_len, args):
        """ model_name = [resnet34, resnet50, densenet161, inceptionresnetv2]  """
        super(Model, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__[model_name](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__[model_name](pretrained=None)

        if args.pretrained_model == "resnet18" or args.pretrained_model == "resnet34":
            self.fc1 = nn.Linear(512, teams_dic_len)  # For Teams class
            self.fc2 = nn.Linear(512, players_dic_len)  # For players class
        elif args.pretrained_model == "densenet161":
            self.fc1 = nn.Linear(2208, teams_dic_len)
            self.fc2 = nn.Linear(2208, players_dic_len)
        elif args.pretrained_model == "inceptionresnetv2":
            self.fc1 = nn.Linear(1536, teams_dic_len)
            self.fc2 = nn.Linear(1536, players_dic_len)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label2 = self.fc2(x)
        return {"label1": label1, "label2": label2}

    def compilation_parameters(self, model_CNN, args):
        # For multilabel output: Same criterion here for 'teams' and 'players', but can change as required
        criterions = {
            phase: nn.CrossEntropyLoss() if phase == "teams" else nn.CrossEntropyLoss()
            for phase in ["teams", "players"]
        }

        if args.optimizer == "adam":
            optimizer = optim.Adam(model_CNN.parameters(), lr=args.learning_rate)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(
                model_CNN.parameters(), lr=args.learning_rate, momentum=0.9
            )
        return criterions, optimizer

    def train_model(
        self, model, dataloaders, criterions, optimizer, device, n_epochs=4
    ):
        """returns trained model"""

        valid_loss_min = np.Inf
        running_loss = {}
        running_loss_record = {"train": [], "valid": []}
        for epoch in range(n_epochs):
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss[phase] = 0.0

                for batch_idx, sample_batched in enumerate(dataloaders[phase]):
                    # importing data and moving to GPU
                    image, label1, label2 = (
                        sample_batched["images"].to(device),
                        sample_batched["teams"].to(device),
                        sample_batched["players"].to(device),
                    )

                    if phase == "train":
                        optimizer.zero_grad()

                    # RuntimeError: expected scalar type Byte but found Float
                    image = image.float()

                    output = model(image)
                    label1_hat = output["label1"]
                    label2_hat = output["label2"]
                    # calculate loss
                    loss1 = criterions["teams"](
                        label1_hat, label1.squeeze().type(torch.LongTensor)
                    )
                    loss2 = criterions["players"](
                        label2_hat, label2.squeeze().type(torch.LongTensor)
                    )
                    loss = loss1 + loss2

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss[phase] = running_loss[phase] + (
                        (1 / (batch_idx + 1)) * (loss.data - running_loss[phase])
                    )

            running_loss_record['train'].append(running_loss["train"])
            running_loss_record['valid'].append(running_loss["valid"])
            print(
                "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                    epoch + 1, running_loss["train"], running_loss["valid"]
                )
            )

            # Save the model if validation loss has decreased
            # if running_loss['valid'] < valid_loss_min:
            #         torch.save(model, "model_cnn.pt")
            #         print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
            #                 valid_loss_min, running_loss['valid']))
            #         valid_loss_min = running_loss['valid']

        plt.plot(range(1, n_epochs+1), running_loss_record['train'])
        plt.plot(range(1, n_epochs+1), running_loss_record['valid'])
        plt.legend(["Training Loss", "Validation Loss"])
        plt.show()
        return model