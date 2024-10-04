
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from omnisafe.utils.riverine_dataset import RiverDataset, get_train_test_datasets

from encoder.vae_v1 import VAE  # Note this is the new version of VAE
from encoder.dataset import InputChannelConfig


"""
Define IO constants
"""
IMG_HEIGHT, IMG_WIDTH = 128, 128

CHANNEL_CONFIG = InputChannelConfig.RGB_MASK

MODEL_SAVE_PATH = f'vae-{CHANNEL_CONFIG.value}channel.pth'

"""
Seed constants
"""
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

"""
Training constants
"""
latent_dim = 64
hidden_dims = [16, 32, 64, 128]
batch_size = 128
epochs = 100
learning_rate = 1e-3


def train_vae(
    train_dataset: RiverDataset,
    model: VAE,
    batch_size: int = 8,
    epochs: int = 200,
):
    """
    Train VAE network for image encoding
    Args:
        train_dataset:
        model:
        batch_size:
        epochs:

    Returns:

    """
    # Set the vae model to train mode
    model.train()

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    print(f'Start training ...')
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            cur_rgb, cur_mask, act, next_rgb, next_mask = data

            # Concat rgb and mask along the channel dimension
            cur_rgb_mask = torch.cat((cur_rgb, cur_mask), dim=1)

            optimizer.zero_grad()

            # Backpropagate loss
            recon, _, mu, logvar = model(cur_rgb_mask)
            loss = model.loss_function(recon, cur_rgb_mask, mu, logvar)['loss']
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {}, Average loss: {:.4f}'.format(epoch, train_loss / len(train_dataset)))
        # if epoch % 10 == 0:
        #     save = to_img(recon_batch.cpu().data)
        #     save_image(save, './vae_img/image_{}.png'.format(epoch))

    print(f'Training finished!')

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved!')


if __name__ == '__main__':
    # Init dataset
    train_dataset, test_dataset = get_train_test_datasets()

    # Init model
    model = VAE(
        in_channels=CHANNEL_CONFIG.value,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        target_output_size=2,
        original_height=IMG_HEIGHT,
        original_width=IMG_WIDTH,
    )

    # Start training
    train_vae(
        train_dataset=train_dataset,
        model=model,
        batch_size=batch_size,
        epochs=epochs,
    )



