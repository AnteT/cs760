from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data_utils

targetdir = './data/mnist'
mnist_trainset = datasets.MNIST(root=targetdir, train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root=targetdir, train=False, download=True, transform=ToTensor())

metrics_dict = {}

class NNScratch():
    def __init__(self, input_shape:int, layers_shape:list, output_shape:int=10, batch_size:int=64):
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.W1 = 2 * torch.rand([layers_shape[0], input_shape]) - 1
        self.B1 = 2 * torch.rand(layers_shape[0]) - 1
        self.W2 = 2 * torch.rand([layers_shape[1], layers_shape[0]]) - 1
        self.B1 = 2 * torch.rand(layers_shape[1]) - 1
        self.W3 = 2 * torch.rand([output_shape, layers_shape[1]]) - 1
        self.B3 = 2 * torch.rand(output_shape) - 1

    def softmax(self, z:np.ndarray) -> np.ndarray:
        calc = torch.exp(z - torch.max(z))
        return calc / torch.sum(calc)
    
    def forward_pass(self, x) -> np.ndarray:
        Z1 = torch.matmul(self.W1, x) + self.B1
        A1 = torch.sigmoid(Z1)
        Z2 = torch.matmul(self.W2, A1) + self.B1
        A2 = torch.sigmoid(Z2)
        Z3 = torch.matmul(self.W3, A2) + self.B3
        ypred = self.softmax(Z3)
        return ypred

    
    def backward_pass(self, x:np.ndarray, y:np.ndarray) -> dict:
        Z1 = torch.matmul(self.W1, x) + self.B1
        A1 = torch.sigmoid(Z1)
        Z2 = torch.matmul(self.W2, A1) + self.B1
        A2 = torch.sigmoid(Z2)
        Z3 = torch.matmul(self.W3, A2) + self.B3
        yhat = self.softmax(Z3)
        batch_loss = -torch.sum(y * torch.log(yhat))
        grad_b3 = yhat - y
        grad_W3 = torch.outer(grad_b3, A2)
        grad_b2 =  torch.matmul(torch.t(self.W3), grad_b3) * A2 * (1-A2) 
        grad_W2 = torch.outer(grad_b2, A1)
        grad_b1 = torch.matmul(torch.t(self.W2), grad_b2) * A1 * (1-A1)
        grad_W1 = torch.outer(grad_b1, x)
        gradient_bias = [grad_b1, grad_b2, grad_b3]
        gradient_weight = [grad_W1, grad_W2, grad_W3]
        return {"gradient_bias": gradient_bias, "gradient_weight" : gradient_weight, "batch_loss": batch_loss}
    

    def batch_descent(self, images:list, labels:list, lr:float, grad:bool=True) -> float:
        images = images.view(images.shape[0], -1)
        W1_gradients = torch.zeros(self.W1.shape)
        B1_gradients = torch.zeros(self.B1.shape)
        W2_gradients = torch.zeros(self.W2.shape)
        B2_gradients = torch.zeros(self.B1.shape)
        W3_gradients = torch.zeros(self.W3.shape)
        B3_gradients = torch.zeros(self.B3.shape)
        batch_loss = 0
        if not grad:
            for (image, label) in zip(images, labels):
                y_one_hot = torch.zeros(self.output_shape)
                y_one_hot[label] = 1
                gradients_dict = self.backward_pass(image, y_one_hot)
                batch_loss += gradients_dict["batch_loss"]
            return batch_loss
        for (image, label) in zip(images, labels):
            y_one_hot = torch.zeros(self.output_shape)
            y_one_hot[label] = 1
            gradients_dict = self.backward_pass(image, y_one_hot)
            batch_loss += gradients_dict["batch_loss"]
            W1_gradients = W1_gradients + gradients_dict["gradient_weight"][0]
            W2_gradients = W2_gradients + gradients_dict["gradient_weight"][1]
            W3_gradients = W3_gradients + gradients_dict["gradient_weight"][2]
            B1_gradients = B1_gradients + gradients_dict["gradient_bias"][0]
            B2_gradients = B2_gradients + gradients_dict["gradient_bias"][1]
            B3_gradients = B3_gradients + gradients_dict["gradient_bias"][2]
        length_x = len(images)
        self.W1 = self.W1 - (lr/length_x * W1_gradients)
        self.W2 = self.W2 - (lr/length_x * W2_gradients)
        self.W3 = self.W3 - (lr/length_x * W3_gradients)
        self.B1 = self.B1 - (lr/length_x * B1_gradients)
        self.B1 = self.B1 - (lr/length_x * B2_gradients)
        self.B3 = self.B3 - (lr/length_x * B3_gradients)
        return batch_loss
    
    def train(self, epochs:int, mnist_trainset:list, batch_size:int, lr:float) -> list:
        total_number_of_inter_epoch_steps = 0
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        for e in range(epochs):
            trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)
            running_loss = 0
            for train_step, (images, labels) in enumerate(trainloader):
                batch_loss = self.batch_descent(images, labels, lr, grad=True)
                running_loss += batch_loss
                if (train_step % 200 == 0) and (train_step != 0): # to get 10 samples from each epoch
                    total_number_of_inter_epoch_steps += 1
                    current_batch_loss_avg = (batch_loss/batch_size).item()
                    current_batch_accuracy_avg = self.get_batched_accuracy(images,labels)
                    test_loss_avg, test_accuracy_avg = self.get_batched_test_metrics()
                    store_metrics(e+1, total_number_of_inter_epoch_steps, current_batch_loss_avg, current_batch_accuracy_avg, test_loss_avg, test_accuracy_avg)
                    print(f"""E{e+1}/{total_number_of_inter_epoch_steps} | train loss: {current_batch_loss_avg:.4f}, train accuracy: {current_batch_accuracy_avg:.4f}, test loss: {test_loss_avg:.4f}, test accuracy: {test_accuracy_avg:.4f}""")
            train_accuracy = self.get_train_accuracy() # removed once batch process implemented
            print(f"Epoch {e+1}/{epochs}:\ttraining loss: {running_loss/(len(trainloader)*batch_size)}, train accuracy: {train_accuracy}")
            train_losses.append(running_loss/(len(trainloader)*batch_size))
            train_accuracies.append(train_accuracy)
            ### get test loss & test accuracy ###
            running_test_loss = 0
            for test_step, (images, labels) in enumerate(testloader):
                test_batch_loss = self.batch_descent(images, labels, lr, grad=False)
                running_test_loss += test_batch_loss
            test_accuracy = self.get_test_accuracy()
            print(f"Epoch {e+1}/{epochs}:\ttest loss: {running_test_loss/(len(testloader)*batch_size)}, test accuracy: {test_accuracy}")
            test_losses.append(running_test_loss/(len(testloader)*batch_size))
            test_accuracies.append(test_accuracy)
        return train_losses, train_accuracies, test_losses, test_accuracies

    def get_batched_test_metrics(self) -> tuple:
        testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=self.batch_size, shuffle=True)
        test_images, test_labels = next(iter(testloader))
        test_batch_loss = self.batch_descent(test_images, test_labels, self.lr, grad=False)
        test_avg_loss = (test_batch_loss/len(test_labels)).item()
        all_count, correct_count = 0, 0
        for i in range(len(test_labels)):
            image = test_images[i].view(784)
            model_output = self.forward_pass(image)
            probabilities = list(model_output.numpy())
            pred_label = probabilities.index(max(probabilities))
            true_label = test_labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
        test_avg_accuracy = correct_count/all_count
        return test_avg_loss, test_avg_accuracy
    
    def get_batched_accuracy(self, images:list, labels:list) -> tuple:
        all_count, correct_count = 0, 0
        for i in range(len(labels)):
            image = images[i].view(784)
            model_output = self.forward_pass(image)
            probabilities = list(model_output.numpy())
            pred_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
        return correct_count/all_count  

    def get_train_accuracy(self) -> tuple:
        trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=self.batch_size, shuffle=True)
        all_count, correct_count = 0, 0
        for images,labels in trainloader:
            for i in range(len(labels)):
                image = images[i].view(784)
                model_output = self.forward_pass(image)
                probabilities = list(model_output.numpy())
                pred_label = probabilities.index(max(probabilities))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1
        return correct_count/all_count    

    def get_test_accuracy(self) -> tuple:
        testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=self.batch_size, shuffle=True)
        all_count, correct_count = 0, 0
        for images,labels in testloader:
            for i in range(len(labels)):
                image = images[i].view(784)
                model_output = self.forward_pass(image)
                probabilities = list(model_output.numpy())
                pred_label = probabilities.index(max(probabilities))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1
        return correct_count/all_count

def store_metrics(epoch:int, step_index:int, train_loss:float, train_accuracy:float, test_loss:float, test_accuracy:float) -> None:
    metric_key = f'{epoch}-{step_index}'
    if metric_key not in metrics_dict.keys():
        metrics_dict[metric_key] = []
    metrics_dict[metric_key].extend([epoch, step_index, train_loss, train_accuracy, test_loss, test_accuracy])


def master_run_training(epochs:int=50, lr:float=0.01) -> None:
    # hyperparams
    batch_size = 64
    input_shape = 784
    layers_shape = [300, 300]
    output_shape = 10
    lr = 0.01
    epochs = 50

    ### train model and report train & test metrics ###
    nnscratch = NNScratch(input_shape=input_shape, layers_shape=layers_shape, output_shape=output_shape)
    train_losses, train_accuracies, test_losses, test_accuracies = nnscratch.train(epochs, mnist_trainset, batch_size, lr)

    ### get final test accuracy ###
    correct_count, all_count = 0, 0
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)

    for images,labels in testloader:
        for i in range(len(labels)):
            image = images[i].view(784)
            model_output = nnscratch.forward_pass(image)
            probabilities = list(model_output.numpy())
            pred_label = probabilities.index(max(probabilities))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    ### pretty print model data for plotting learning curve ###
    print(f"""
    {'-'*140}
    hyperparameters & training results:
    batch_size: {batch_size}
    input_shape: {input_shape}
    layers_shape: {layers_shape}
    output_shape: {output_shape}
    learning_rate: {lr}
    epochs: {epochs}
    validation_size: {all_count} images
    accuracy: {correct_count/all_count}
    loss_data = (""")
    for e,loss in enumerate(train_losses):
        if e == 0:
            print(f"({e+1}, {loss.item():.4f}, {train_accuracies[e]}, {test_losses[e].item():.4f}, {test_accuracies[e]})")
        else:
            print(f",({e+1}, {loss.item():.4f}, {train_accuracies[e]}, {test_losses[e].item():.4f}, {test_accuracies[e]})")
    print(f")\n{'-'*140}")
    print(f"### batched metrics ###\nbatch_headers = ['epoch', 'step', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']\nbatch_data = [")
    for i,v in enumerate(metrics_dict.values()):
        if i == 0:
            print(f' {v}')
        else:
            print(f',{v}')
    print(f"]\n{'-'*140}")
    print('finished')

########################### run main model ###########################
if __name__ == '__main__':
    master_run_training(epochs=50, lr=0.01)
