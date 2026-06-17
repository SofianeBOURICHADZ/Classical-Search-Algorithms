import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models,transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self,noise_vector_size,feature_g):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(noise_vector_size,feature_g * 16 ,kernel_size=4,stride=1,padding = 0,bias = False),
            nn.BatchNorm2d(feature_g * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_g * 16, feature_g * 8,kernel_size=4,padding=1,stride = 2,bias = False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4,kernel_size=4,padding=1,stride = 2,bias = False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2,kernel_size=4,padding=1,stride = 2,bias = False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_g * 2, 3 ,kernel_size=4,padding=1,stride = 2,bias = False),
            nn.Tanh(),
        )
    def forward(self,x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self,feature_dim):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3,feature_dim,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim,feature_dim * 2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim*2,feature_dim * 4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim*4,feature_dim * 8,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self,x):
        return (self.extractor(x).view(-1))
    
generator = Generator(100,256)
discriminator = Discriminator(64)

gen_optim = optim.AdamW(generator.parameters(),lr = 0.01,weight_decay=0.0001)
disc_optim = optim.AdamW(discriminator.parameters(),lr = 0.01,weight_decay=0.0001)
criterion = nn.BCEWithLogitsLoss()

def train_one_epoch(dataloader,device):
    generator.train()
    discriminator.train()
    total = 0
    running_loss_gen = 0.0
    running_loss_di = 0.0
    bar = tqdm(dataloader,'Training')
    for img in bar:
        img = img.to(device)
        disc_optim.zero_grad()
        gen_optim.zero_grad()
        noise_vector = torch.rand((32,100,1,1),dtype=torch.float32)
        results_real = discriminator(img)
        real_loss = criterion(results_real,torch.ones_like(results_real))
        datafake = generator(noise_vector.detach())
        fake_loss = criterion(datafake,torch.zeros_like(fake_loss))
        racist_loss = real_loss+fake_loss
        racist_loss.backward()
        disc_optim.step()
        gen_fake_out = discriminator(datafake)
        gen_loss = criterion(gen_fake_out,torch.ones_like(gen_fake_out))
        gen_loss.backward()
        gen_optim.step()
        running_loss_gen += gen_loss.item()
        running_loss_di += racist_loss.item()
        total += img.shape[0]
        bar.set_postfix(LossGen = gen_loss,LossDisc = racist_loss)
    return running_loss_di / total,running_loss_gen / total

for i in range(0,5):
    print(f'EPoch {i}')
    dl,gl = train_one_epoch(device=device)