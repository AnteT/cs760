import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import imageio

# learning parameters
output_suffix = '-x' # use ('-a', '-b', '-x') in combination with standard disc/gen flags to trigger various implementation
standard_discriminator = False # triggers discriminator presets for output suffix implementation above
standard_generator = False # triggers generator presets for output suffix implementation above
k_step = 1
lr = 2e-4
visualize_training_images = False
batch_size = 512
epochs = 100
sample_size = 64 # fixed sample size for generator
nz = 128 # latent vector size # TODO changed from 128 > 256
k_dis = k_step # number of steps discriminator
k_gen = k_step # number of steps generator (if applies)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])
to_pil_image = transforms.ToPILImage()
results_dict = {}

# Load train data
train_data = datasets.MNIST(
    root='input/data',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)
print('##### GENERATOR #####')
print(generator)
print('######################')
print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr)
optim_d = optim.Adam(discriminator.parameters(), lr=lr)

# loss function
criterion = nn.BCELoss() # Binary Cross Entropy loss
losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch
images = [] # to store images generatd by the generator

# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)

# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)

# function to create the noise vector
def create_noise(sample_size, nz):
    v = torch.randn(sample_size, nz).to(device)
    return v

# to save the images generated by the generator
def save_generator_image(image, path):
    save_image(image, path)

# create the noise vector - fixed to track how GAN is trained.
noise = create_noise(sample_size, nz)

torch.manual_seed(7777)

def generator_loss_standard(fake_output, sample_size):
    ############ YOUR CODE HERE ##########
    loss_g = criterion(fake_output, label_real(sample_size))
    return loss_g
    ######################################

def generator_loss_2(fake_output, sample_size):
    ############ YOUR CODE HERE ##########
    loss_g = - criterion(fake_output, label_fake(sample_size))
    return loss_g
    ######################################

def discriminator_loss_standard(output, true_label):
    ############ YOUR CODE HERE ##########
    return criterion(output, true_label)
    ######################################

def discriminator_loss_2(fake_output, fake_label, real_output, real_label):
    ############ YOUR CODE HERE ##########
    loss_d = criterion(fake_output, fake_label) + criterion(real_output, real_label)
    return loss_d
    ######################################

### training loop begins ###
print(f"{'-'*130}\nSTARTING TRAINING\n{'-'*130}")
for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
        # get the data
        real, _ = data
        real = real.to(device)
        # train the discriminator
        for _ in range(k_dis):
            # generate fake images
            if standard_discriminator:
                fake = generator(create_noise(sample_size, nz))
                # get the loss for the discriminator
                fake_images = generator(create_noise(sample_size, nz)).detach()
                fake_labels = label_fake(fake_images.shape[0]).to(device)
                fake_output = discriminator(fake_images)
                loss_d = discriminator_loss_standard(fake_output, fake_labels)
            else:
                loss_d = discriminator_loss_2(discriminator(fake), label_fake(sample_size), discriminator(real), label_real(real.size(0)))
            # optimize the discriminator
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()
    
        # train the generator
        for _ in range(k_gen):
            # generate fake images
            fake = generator(create_noise(sample_size, nz))
            # get the loss for the generator
            if standard_generator:
                loss_g = generator_loss_standard(discriminator(fake), sample_size)
            else:
                loss_g = generator_loss_2(discriminator(fake), sample_size)
            # optimize the generator
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()
    
    # create the final fake image for the epoch
    generated_img = generator(noise).cpu().detach()
    
    # make the images as grid
    generated_img = make_grid(generated_img)
    
    # visualize generated images
    if visualize_training_images:
        if (epoch + 1) % 5 == 0:
            plt.imshow(generated_img.permute(1, 2, 0))
            plt.title(f'epoch {epoch+1}')
            plt.axis('off')
            plt.show()
    
    # save the generated torch tensor models to disk
    if epoch in (0,49,99):
        save_generator_image(generated_img, f"outputs/gen_img{epoch+1}{output_suffix}.png")
    images.append(generated_img)
    epoch_loss_g = loss_g # total generator loss for the epoch
    epoch_loss_d = loss_d # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    
    print(f"{epoch+1}|{epochs}: generator loss: {epoch_loss_g:.8f}, discriminator loss: {epoch_loss_d:.8f}")
    results_dict[epoch+1] = (epoch+1, round(epoch_loss_g.item(),6), round(epoch_loss_d.item(),6))

print(f"{'-'*130}\nDONE TRAINING\n{'-'*130}")
torch.save(generator.state_dict(), f'outputs/generator{output_suffix}.pth')
imgs = [np.array(to_pil_image(img)) for img in images]
imageio.mimsave(f'outputs/generator_images{output_suffix}.gif', imgs)

for i in range(len(losses_g)):
    losses_g[i] = np.array(losses_g[i].cpu().detach().numpy())
    
for i in range(len(losses_d)):
    losses_d[i] = np.array(losses_d[i].cpu().detach().numpy())

# plot and save the generator and discriminator loss
plt.figure()
plt.plot(losses_g, label='Generator loss')
plt.plot(losses_d, label='Discriminator Loss')
plt.legend()
plt.savefig(f'outputs/loss{output_suffix}.png')
for row in results_dict.values():
    if row[0] == 1:
        print("training_headers = ('epoch', 'generator_loss', 'discriminator_loss')")
        print(f"training_results = (")
        print(row)
    else:
        print(f",{row}")
print(")")
plt.show()