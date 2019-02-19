import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils

ENTROPY_THRESHOLD = 20
N_VALIDATE = 1000
N_CLASSES = 10

class Trainer(object):

    def __init__(self, model, trainloader, testloader, n_epoch, n_sampling, lr):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.n_epoch = n_epoch
        self.n_sampling = n_sampling
        self.optim = optim.SGD(self.model.parameters(), lr=lr)

    def train(self):
        for epoch in range(self.n_epoch):
            self.model.train()
            for i, (images, labels) in enumerate(self.trainloader):
                self.optim.zero_grad()
                preds = self.model(images)
                loss = F.nll_loss(torch.log(preds), labels)

                loss.backward()

                self.optim.step()
            print("Epoch: {} Loss: {}".format(epoch, loss))

    def validate(self):
        self.model.train()
        entropy = torch.zeros(N_VALIDATE)
        preds = torch.zeros(self.n_sampling, N_CLASSES)
        images = []
        for i, (image, labels) in enumerate(self.testloader):
            if (i + 1) > N_VALIDATE:
                break
            for sampling in range(self.n_sampling):
                preds[sampling, :] = self.model(image).squeeze()

            pred = preds.mean(dim=0)
            entropy[i] = torch.sum(- pred * torch.log(pred), dim=0)
            images.append(image.squeeze().unsqueeze(0))

        images = torch.stack(images)
        _, sort_idx = entropy.sort()
        high_entropy_images = images[sort_idx][:ENTROPY_THRESHOLD]
        vutils.save_image(high_entropy_images, "high_entropy.jpg", nrow=4)
        low_entropy_images = images[sort_idx][-ENTROPY_THRESHOLD:]
        vutils.save_image(low_entropy_images, "low_entropy.jpg", nrow=4)
