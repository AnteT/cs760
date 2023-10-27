from typing import Self
import pickle, os, sys, datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

bs = 64
targetdir = './data/mnist'
mnist_trainset = datasets.MNIST(root=targetdir, train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root=targetdir, train=False, download=True, transform=ToTensor())
trainloader = DataLoader(mnist_trainset, batch_size=bs, shuffle=True)
testloader = DataLoader(mnist_testset, batch_size=bs, shuffle=True)

def get_now_dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def peek_mnist_images():
    images, labels = next(iter(trainloader))
    print(f"images shape: {images.shape}")
    print(f"labels shape: {labels.shape}")
    figure = plt.figure()
    num_of_images = 30
    for index in range(1, 1+num_of_images):
        plt.subplot(3, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.show()

def get_training_sample():
    sample_idx = torch.randint(len(mnist_trainset), size=(1,)).item()
    img, label = mnist_trainset[sample_idx]
    return img, label

def get_training_batch(batch_size:int=32):
    trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    x_train, y_train = next(iter(trainloader))
    return x_train, y_train

class NNExperimental(nn.Module):
    def __init__(self, input_layer:int=784, hidden_layer:int = 300, output_layer:int=10, lr:float = 0.01, batch_size:int = 64, epochs:int=50):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.flatten = nn.Flatten()
        self.W1 = torch.tensor(np.random.randn(input_layer, hidden_layer) / np.sqrt(hidden_layer), requires_grad=True, dtype=torch.float32)
        self.B1 = torch.tensor(np.random.uniform(-1.0,1.0,(1,hidden_layer)), requires_grad=True, dtype=torch.float32)
        self.A1 = nn.Sigmoid()
        self.W2 = torch.tensor(np.random.randn(hidden_layer, output_layer) / np.sqrt(output_layer), requires_grad=True, dtype=torch.float32)
        self.B2 = torch.tensor(np.random.uniform(-1.0,1.0,(1,output_layer)), requires_grad=True, dtype=torch.float32)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def to_pickle(self, filename:str=None) -> None:
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.ptexp'
        params_dict = {'W1': self.W1,'B1': self.B1,'W2': self.W2,'B2': self.B2, 'results_dict':self.results_dict}
        with open(filename, 'wb') as f:
            pickle.dump(params_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'successfully saved model params as {filename}')
    
    def from_pickle(self, filename:str = None) -> None:
        if filename is None:
            filename = os.path.basename(sys.argv[0]).split(".")[0]+'.ptexp'
        with open(filename, 'rb') as f:
            params_dict = pickle.load(f)
        self.W1 = params_dict['W1']
        self.B1 = params_dict['B1']
        self.W2 = params_dict['W2']
        self.B2 = params_dict['B2']
        self.results_dict = params_dict['results_dict']
        print(f'successfully loaded model params from {filename}')
        
    def forward(self, x):
        x = self.flatten(x) # converts to shape: (n, 784)
        x = torch.matmul(x, self.W1) + self.B1 # shape: (n, 300)
        x = self.A1(x) # first and only sigmoid activation
        x = torch.matmul(x, self.W2) + self.B2 # shape: (n, 10)
        return x
    
    def loss_fn(self, y_pred, y_true):
        return self.cross_entropy_loss(y_pred,y_true)
    
    def get_batch_metrics(self):
        ### get training metrics ###
        trainloader = DataLoader(mnist_trainset, batch_size=self.batch_size, shuffle=True)
        with torch.no_grad():
            train_x, train_y = next(iter(trainloader))
            train_total_num, train_correct_num = len(train_y), 0
            train_predictions = self(train_x)
            train_correct_num += (train_predictions.argmax(1) == train_y).type(torch.float).sum().item()
            train_avg_loss = (self.loss_fn(train_predictions, train_y).item())
            train_avg_accuracy = train_correct_num/train_total_num
        
        ### get test metrics ###
        testloader = DataLoader(mnist_testset, batch_size=self.batch_size, shuffle=True)
        with torch.no_grad():
            test_x, test_y = next(iter(testloader))
            test_total_num, test_correct_num = len(test_y), 0
            test_predictions = self(test_x)
            test_correct_num += (test_predictions.argmax(1) == test_y).type(torch.float).sum().item()
            test_avg_loss = (self.loss_fn(test_predictions, test_y).item())
            test_avg_accuracy = test_correct_num/test_total_num        
        
        return train_avg_loss, train_avg_accuracy, test_avg_loss, test_avg_accuracy
    
    def full_test_accuracy(self):
        num_batches = len(testloader)
        size = len(testloader.dataset)
        test_loss, num_correct = 0, 0
        with torch.no_grad():
            for ib, (xb, yb) in enumerate(testloader):
                preds = self(xb)
                test_loss += self.loss_fn(preds, yb).item()
                if ib % 100 == 0:
                    print(f"y_pred: {preds.argmax(1)[:10]}\ny_true: {yb[:10]}\n({int((preds.argmax(1) == yb).type(torch.float).sum().item())}/{self.batch_size} correct)")
                num_correct += (preds.argmax(1) == yb).type(torch.float).sum().item()
        test_loss = test_loss / num_batches
        test_accuracy = num_correct / size
        print(f"Model test loss: {test_loss:.4f}, model test accuracy: {test_accuracy:.2%} ({int(num_correct)} of {size} images)")

    def display_training_data(self):
        print(f"### batched metrics for bs {self.batch_size} using lr {self.lr} after {self.epochs} epochs ###\nbatch_headers = ['epoch', 'step', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']\nbatch_data = [")
        for i,v in enumerate(self.results_dict.values()): 
            if i == 0:
                print(f' {v}')
            else:
                print(f',{v}')
        print(f"]\n{'-'*140}")     

    ### setup training loop ###
    def train_model(self, batch_size:int=64, lr:float=0.01, epochs:int=10):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.results_dict = {}
        total_batch_iterations, report_steps = 0, 0
        trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for x_batch, y_batch in trainloader:
                total_batch_iterations += 1
                y_pred = self(x_batch)
                loss = self.cross_entropy_loss(y_pred, y_batch)
                loss.backward()
                # batch_loss = loss.item() / batch_size # unused
                self.W1.data = self.W1.data - (lr * self.W1.grad.data)
                self.B1.data = self.B1.data - (lr * self.B1.grad.data)
                self.W2.data = self.W2.data - (lr * self.W2.grad.data)
                self.B2.data = self.B2.data - (lr * self.B2.grad.data)
                self.W1.grad.data.zero_()
                self.B1.grad.data.zero_()
                self.W2.grad.data.zero_()
                self.B2.grad.data.zero_()
                if (total_batch_iterations % 200 == 0) and (total_batch_iterations > 0):
                    report_steps += 1
                    train_loss, train_acc, test_loss, test_acc = self.get_batch_metrics()
                    self.results_dict[report_steps] = [(epoch+1), report_steps, train_loss, train_acc, test_loss, test_acc]
                    display_prefix = f"""{epoch+1}|{total_batch_iterations}"""
                    print(f'{display_prefix:-<12}{train_loss:.4f} train loss, {train_acc:.4f} train accuracy, {test_loss:.4f} test loss, {test_acc:.4f} test accuracy')
        return self.results_dict

def load_model_from_ptexp(modelpath:str=None) -> NNExperimental:
    ### load and return model ###
    loaded_model = NNExperimental()
    if modelpath is None:
        loaded_model.from_pickle()
    else:
        loaded_model.from_pickle(modelpath)
    return loaded_model

def master_run_training(epochs:int=50, lr:float=0.01, pickle_params:bool=True):
    ### instantiate model ###
    print('-'*140)
    print(f'{get_now_dt()} started training...')
    NN = NNExperimental()

    ### train model ###
    NN.train_model(epochs=epochs,lr=lr)
    NN.display_training_data()
    NN.full_test_accuracy()

    ### save & test load model ###
    if pickle_params:
        NN.to_pickle()
        NN.from_pickle()
        NN.full_test_accuracy()
    
    ### finish ###
    print(f'{get_now_dt()} finished training')
    print('-'*140)

########################### run experimental model ###########################
if __name__ == '__main__':
    master_run_training(epochs=200, lr=0.02, pickle_params=True)
    NN = load_model_from_ptexp() # load from ptexp file
    NN.full_test_accuracy() # confirm successful load
