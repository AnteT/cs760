### pytorch version for mnist dataset ###
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import onnx

targetdir = './data/mnist'
training_data = datasets.MNIST(root=targetdir, train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root=targetdir, train=False, download=True, transform=ToTensor())

def get_random_training_sample():
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    image, label = training_data[sample_idx]
    return image, label

class NeuralNet(nn.Module):
    def __init__(self, lr:float=1e-2):
        super().__init__()
        self.lr = lr
        self.batch_size = None
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(784,300)
            ,nn.Sigmoid()
            ,nn.Linear(300,10)
        )
        self.loss_fn = nn.CrossEntropyLoss() # using cross entropy loss
        self.optimizer = torch.optim.SGD(self.parameters(),lr=lr)
        self.batch_sample_counter = None
        self.training_results = {}
        self.results_dict = {}

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

    def train_model(self, batch_size:int = 32, e_current:int=None):
        self.batch_size = batch_size
        trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        if self.batch_sample_counter is None:
            self.batch_sample_counter = 0
        training_loss = 0
        for train_step, (xb, yb) in enumerate(trainloader):
            preds = self(xb)
            loss = self.loss_fn(preds, yb)
            training_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (train_step % 200 == 0) and (train_step != 0):
                self.batch_sample_counter += 1
                tr_loss, tr_accuracy, ts_loss, ts_accuracy = self.get_batch_metrics()
                self.results_dict[self.batch_sample_counter] = [e_current+1, self.batch_sample_counter, tr_loss, tr_accuracy, ts_loss, ts_accuracy]
                print(f'E{e_current+1}/{self.batch_sample_counter} | tr_loss: {tr_loss:.4f}, tr_accuracy: {tr_accuracy:.4f}, ts_loss: {ts_loss:.4f}, ts_accuracy: {ts_accuracy:.4f}')
        training_loss_avg = training_loss/(len(trainloader))
        ### get training accuracy ###
        with torch.no_grad():
            num_total, num_correct = len(trainloader.dataset), 0
            for xb, yb in trainloader:
                preds = self(xb)
                num_correct += (preds.argmax(1) == yb).type(torch.float).sum().item()
        train_accuracy = num_correct/num_total
        self.training_results[e_current] = [f'{training_loss_avg:.4f}', f'{train_accuracy:.4f}']
        print(f"Epoch: {e_current+1} | Training loss avg: {training_loss_avg:.4f}, Training Accuracy: {train_accuracy:.4f}")

    def get_batch_metrics(self):
        ### get training metrics ###
        trainloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
        with torch.no_grad():
            train_x, train_y = next(iter(trainloader))
            train_total_num, train_correct_num = len(train_y), 0
            train_predictions = self(train_x)
            train_correct_num += (train_predictions.argmax(1) == train_y).type(torch.float).sum().item()
            train_avg_loss = (self.loss_fn(train_predictions, train_y).item())
            train_avg_accuracy = train_correct_num/train_total_num
        
        ### get test metrics ###
        testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        with torch.no_grad():
            test_x, test_y = next(iter(testloader))
            test_total_num, test_correct_num = len(test_y), 0
            test_predictions = self(test_x)
            test_correct_num += (test_predictions.argmax(1) == test_y).type(torch.float).sum().item()
            test_avg_loss = (self.loss_fn(test_predictions, test_y).item())
            test_avg_accuracy = test_correct_num/test_total_num        
        return train_avg_loss, train_avg_accuracy, test_avg_loss, test_avg_accuracy
                
    def test_model(self, batch_size:int=32, e_current:int=None):
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        num_batches = len(testloader)
        size = len(testloader.dataset)
        test_loss, corrects = 0, 0
        with torch.no_grad():
            for xb, yb in testloader:
                preds = self(xb)
                test_loss += self.loss_fn(preds, yb).item()
                corrects += (preds.argmax(1) == yb).type(torch.float).sum().item()
        test_loss /= num_batches
        # test_loss = lo
        corrects /= size
        # self.training_results[e_current].extend(['{:.4f}'.format(test_loss),'{:.2f}'.format(100*corrects)])
        self.training_results[e_current].extend([f'{test_loss:.4f}',f'{corrects:.4f}'])
        print(f"Test average loss: {test_loss:.4f}, Test accuracy: {corrects:.4f}")

def init_weights_zeroed(model):
    if isinstance(model. nn.Linear):
        nn.init.zeros_(model.weight) # zero all weights and biases
        model.bias.data.fill_(0)

def init_weights_uniform(model):
    if isinstance(model. nn.Linear):
        nn.init.uniform_(model.weight,-1.0,1.0) # fill weights according to the passed function

def init_weights_xavier(model):
    if isinstance(model. nn.Linear):
        nn.init.xavier_uniform_(model.weight) # xavier uniform weight initialization

def run_full_training_cycle(lr:float=1e-1, batch_size:int=32, epochs:int=10, modify_weights:str=None, savemodel:str=None, saveonnx:str=None):
    NN = NeuralNet(lr=lr)
    print(NN)
    if modify_weights is not None:
        if modify_weights == 'zeroed':
            NN.apply(init_weights_zeroed)
        if modify_weights == 'uniform':
            NN.apply(init_weights_uniform)
        else:
            NN.apply(init_weights_xavier)
        for name, param in NN.named_parameters():
            print(f'name: {name}, param: {param}')
    for e in range(epochs):
        NN.train_model(batch_size=batch_size, e_current=e)
        NN.test_model(batch_size=batch_size, e_current=e)
    print(f"""### hyperparams: lr = {lr}, epochs = {epochs}, batch_size = {batch_size} ###\nloss_headers = ('epoch', 'training_loss', 'training_accuracy','test_loss', 'test_accuracy')\nloss_data = (""")
    for epoch, results in NN.training_results.items():
        if epoch == 0:
            print(f"({epoch+1}, {', '.join([x for x in results])})")
        else:
            print(f",({epoch+1}, {', '.join([x for x in results])})")
    print(f')')
    batch_data_var_name = None
    try:
        lr_str = str(lr).split('.')[-1]
        bs_str = str(batch_size)
        batch_data_var_name = f"batch_data_lr_{lr_str}_bs_{bs_str}_torch"
    except:
        batch_data_var_name = 'batch_data'
    print(f"### batched metrics ###\nbatch_headers = ['epoch', 'step', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']\n{batch_data_var_name} = [")
    for i,v in enumerate(NN.results_dict.values()): 
        if i == 0:
            print(f' {v}')
        else:
            print(f',{v}')
    print(f"]\n{'-'*140}")
    if savemodel is not None:
        print('saving model state dict...')
        torch.save(NN.state_dict(), savemodel)
        print(f'successfully saved model to path {savemodel}')
    if saveonnx is not None:
        X_test, _ = get_random_training_sample()
        print('exporting model to onyx file...')
        torch.onnx.export(NN, X_test, saveonnx, input_names=["mnist_features"], output_names=["logits"])
    print('finished')

########################### run main model ###########################
if __name__ == '__main__':
    run_full_training_cycle(lr=0.01, batch_size=64, epochs=50, modify_weights= None, savemodel='./models/lr01-bs64-e50.pt', saveonnx='./data/model_base.onnx')
    run_full_training_cycle(lr=0.01, batch_size=64, epochs=50, modify_weights='zeroed', savemodel='./models/lr01-bs64-e50_zeroed.pt', saveonnx='./data/model_zeroed.onnx')
    run_full_training_cycle(lr=0.01, batch_size=64, epochs=50, modify_weights='uniform', savemodel='./models/lr01-bs64-e50_uniform.pt', saveonnx='./data/model_uniform.onnx')

