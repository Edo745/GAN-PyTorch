import torch
import argparse
import matplotlib.pyplot as plt
from models.generator import Generator
import os

def generate_images(checkpoint_path):
    # Set up the generator
    netG = Generator()
    netG.load_state_dict(torch.load(checkpoint_path))
    netG.eval()

    # Set up other parameters
    n = 10
    batch_size = n**2
    latent_size = 100  # Replace with the actual size of your latent space (nz)

    # Generate images
    fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
    fake_images = netG(fixed_noise)

    # Plot
    fake_images_np = fake_images.cpu().detach().numpy()
    fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
    R, C = n, n

    plt.figure(figsize=(8, 8))

    for i in range(batch_size):
        plt.subplot(R, C, i + 1)
        plt.imshow(fake_images_np[i], cmap='gray')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
    os.makedirs("GAN/results/", exist_ok=True)
    plt.savefig("GAN/results/generated_images.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained GAN generator.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the generator checkpoint file.")
    args = parser.parse_args()

    generate_images(args.checkpoint)
