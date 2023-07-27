import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_epochs = 100
latent_dim = 100
image_size = 64
output_dir = "generated_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size * image_size * 3),
            nn.Tanh()  # Tanh activation for image pixels in range [-1, 1]
        )

    def forward(self, z):
        generated_image = self.model(z)
        generated_image = generated_image.view(generated_image.size(0), 3, image_size, image_size)
        return generated_image

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity

# Initialize the models and move them to the device
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define the loss function and optimizer for GAN training
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the loss function and optimizer for GAN training
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Prepare the data loader
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize images to the specified size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


dataset = datasets.ImageFolder(root="/content/Images", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Adversarial ground truths (1 for real, 0 for fake)
        valid = torch.ones(real_images.size(0), 1).to(device)
        fake = torch.zeros(real_images.size(0), 1).to(device)

        # Move real images to the device
        real_images = real_images.to(device)

        # Train the Generator
        optimizer_G.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim).to(device)
        generated_images = generator(z)
        g_loss = adversarial_loss(discriminator(generated_images), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train the Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # Print progress and save generated images
        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )
            save_image(
                generated_images.data[:25],
                f"{output_dir}/generated_{batches_done}.png",
                nrow=5,
                normalize=True,
            )
