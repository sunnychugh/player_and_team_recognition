import pretrainedmodels
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

class Model(nn.Module):
    """Generate the Pytorch model"""
    def __init__(
        self, pretrained, model_name, teams_dic_len, players_dic_len, data, args
    ):
        """
        Parameters:
            pretrained (bool): Take the initial model with pretrained layers or train from scratch.
            model_name: Name of the pretrained model to start with.
            teams_dic_len (int): Number of unique teams present in the dataset.
            players_dic_len (int): Number of unique players present in the dataset.
            data: Train and validation set images to train the model as per requirement.
            args (Argparse): Various initial arguments.
        """
        super(Model, self).__init__()

        self.data = data
        self.dropout = 0.0

        if pretrained is True:
            self.model = pretrainedmodels.__dict__[model_name](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__[model_name](pretrained=None)

        if args.pretrained_model == "resnet18" or args.pretrained_model == "resnet34":
            # For Teams class
            self.fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, teams_dic_len))
            # For players class
            self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, players_dic_len))
        elif args.pretrained_model == "resnet50":
            self.fc1 = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, teams_dic_len))
            self.fc2 = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, players_dic_len))
        elif args.pretrained_model == "densenet161":
            self.fc1 = nn.Sequential(nn.Linear(2208, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, teams_dic_len))
            self.fc2 = nn.Sequential(nn.Linear(2208, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, players_dic_len))
        elif args.pretrained_model == "inceptionresnetv2":
            self.fc1 = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, teams_dic_len))
            self.fc2 = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(self.dropout),nn.Linear(256, players_dic_len))

    def forward(self, x):
        """Defines the labels"""
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label2 = self.fc2(x)
        return {"label1": label1, "label2": label2}

    def compilation_parameters(self, model_CNN, args):
        """
        Defines the compilation parameters.
            Parameters:
                model_CNN (Pytorch model): Model with the proposed layer combination to
                    train with the images dataset.
                args (Argparse): Various initial arguments.
            Returns:
                criterions: Returns the criterion used.
                optimizer: Returns the defined optimizer with parameters values.
        """
        # For multilabel output: Chosing same criterion here for 'teams' and 'players',
        # but can change as required
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
        self,
        model,
        dataloaders,
        criterions,
        optimizer,
        device,
        n_epochs=4,
        show_plot=False,
    ):
        """
        Train the image classification model
            Parameters:
                model (Pytorch model): Model with the proposed layer combination to
                    train with the images dataset.
                dataloaders: Pytorch dataloaders to train the model in batches.
                criterions: Defines the type of loss to be considered.
                optimizer: Defines the type of optimizer considered.
                device: Defines the device (CPU or GPU) being used.
                n_epochs (int): Number of epochs/iterations to be considered.
                show_plot(bool): Show the plot of training and validation accuracy
                    if show_plot is True.
            Returns:
                model (Pythorch): Returns the trained Pytorch model on the images dataset.
        """

        running_corrects = {}
        running_accuracy = {}
        running_accuracy_record = {"train": [], "valid": []}
        for epoch in range(n_epochs):
            since = time.time()
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_accuracy[phase] = 0.0
                running_corrects[phase] = 0.0

                for batch_idx, sample_batched in enumerate(dataloaders[phase]):
                    # importing data and moving to device (GPU, if available)
                    image, label1, label2 = (
                        sample_batched["images"].to(device),
                        sample_batched["teams"].to(device),
                        sample_batched["players"].to(device),
                    )

                    if phase == "train":
                        optimizer.zero_grad()

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

                    # Calculate accuracy
                    _, pred_label1 = torch.max(output["label1"], 1)
                    _, pred_label2 = torch.max(output["label2"], 1)
                    equals = (pred_label1 == label1) & (pred_label2 == label2)
                    running_corrects[phase] += torch.mean(equals.type(torch.FloatTensor)).item()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_accuracy[phase] = running_corrects[phase]/len(dataloaders[phase])

            running_accuracy_record["train"].append(running_accuracy["train"])
            running_accuracy_record["valid"].append(running_accuracy["valid"])
            print(
                "Epoch: {} \tTraining Acc: {:.4f} \tValidation Acc: {:.4f}".format(
                    epoch + 1,
                    running_accuracy["train"],
                    running_accuracy["valid"],
                )
            )
            print(f"time for epoch: {round((time.time() - since) / 60, 2)} mins")

        if show_plot:
            plt.plot(range(1, n_epochs + 1), running_accuracy_record["train"])
            plt.plot(range(1, n_epochs + 1), running_accuracy_record["valid"])
            plt.xlabel("Epochs")
            plt.legend(["Training Accuracy", "Validation Accuracy"])
            plt.show()

        return model
