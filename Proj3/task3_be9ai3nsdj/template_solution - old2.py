from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

"""
README FIRST

The below code is a template for the solution. You can change the code according
to your preferences, but the testing function has to save the output of your 
model on the test data as it does in this template. This output must be submitted.

Replace the dummy code with your own code in the TODO sections.

We also encourage you to use tensorboard or wandb to log the training process
and the performance of your model. This will help you to debug your model and
to understand how it is performing. But the template does not include this
functionality.
Link for wandb:
https://docs.wandb.ai/quickstart/
Link for tensorboard: 
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")

# If you have a Mac consult the following link:
# https://pytorch.org/docs/stable/notes/mps.html

# It is important that your model and all data are on the same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    where N_train_samples is the number of training samples, N_test_samples is
    the number of test samples, C is the number of channels (1 for grayscale),
    H is the height of the image, and W is the width of the image.
    """
    # Load the training data
    train_data = np.load("train_data.npz")["data"]

    # Make the training data a tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)

    # Load the test data
    test_data_input = np.load("test_data.npz")["data"]

    # Make the test data a tensor
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    ########################################
    # TODO: Given the original training images, create the input images and the
    # label images to train your model. 
    # Replace the two placholder lines below (which currently just copy the
    # training data) with your own implementation.
    # what was here
    # train_data_label = train_data.clone()
    # train_data_input = train_data.clone()

    train_data_input = train_data.clone()
    train_data_input[:, :, 10:18, 10:18] = 0  # masked image is the input
    train_data_label = train_data[:, :, :, :]   # clean image is the lable
    

    #train_data_input = train_data_input.float()/255
    #train_data_label = train_data_label.float()/255

    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if False:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze(), cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")

            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input


def training(train_data_input, train_data_label, **kwargs):
    """
    Train the model. Fill in the details of the data loader, the loss function,
    the optimizer, and the training loop.

    Args:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - model: torch.nn.Module
    """
    model = Model()
    model.train()
    model.to(device)

    # TODO: Dummy criterion - change this to the correct loss function
    # https://pytorch.org/docs/stable/nn.html#loss-functions
    # criterion = lambda x, y: torch.mean((x))
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    # TODO: Dummy optimizer - change this to a more suitable optimizer
    # optimizer = torch.optim.SGD(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # TODO: Correctly setup the dataloader - the below is just a placeholder
    # Also consider that you might not want to use the entire dataset for
    # training alone
    # (batch_size needs to be changed)
    batch_size = 64
    dataset = TensorDataset(train_data_input, train_data_label)
    # Consider the shuffle parameter and other parameters of the DataLoader
    # class (see
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    # TODO: Modify the training loop in case you need to

    # TODO: The value of n_epochs is just a placeholder and likely needs to be
    # changed
    n_epochs = 60

    for epoch in range(n_epochs):
        # added stuff
        #epoch_loss = 0

        for x, y in tqdm(
            data_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # added stuff
            #epoch_loss += loss.item()

        # added stuff
        #avg_loss = epoch_loss / len(data_loader)
        #scheduler.step(avg_loss)

        print(f"Epoch {epoch} loss: {loss.item()}")

    return model


# TODO: define a model. Here, a basic MLP model is defined. You can completely
# change this model - and are encouraged to do so.
class Model(nn.Module):
    """
    Implement your model here.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        #self.fc = nn.Linear(784, 784)
    
        # added stuff
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #nn.MaxPool2d(2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            #nn.MaxPool2d(2)
        )

 
        """ self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.up0 = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=True)
        self.dec0 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        ) """

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            #nn.Sigmoid()
        )


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # Flatten the image in the last two dimensions
        #x = x.view(x.shape[0], -1)
        #x = self.fc(x)
        #x = F.relu(x)

        # added stuff
        orig = x.clone()

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        """
        e3 = self.encoder3(e2)

        d0 = self.up0(e3)
        d0 = torch.cat([d0, e2], dim=1)
        d0 = self.dec0(d0)
        """

        d1 = self.up1(e2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, orig], dim=1)
        x = self.dec2(d2)

        # Reshape the image to the original shape
        x = x.view(x.shape[0], 1, 28, 28)
        return x


def testing(model, test_data_input):
    """
    Uses your model to predict the ouputs for the test data. Saves the outputs
    as a binary file. This file needs to be submitted. This function does not
    need to be modified except for setting the batch_size value. If you choose
    to modify it otherwise, please ensure that the generating and saving of the
    output data is not modified.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise to avoid memory issues
        test_data_output = []
        # TODO: You can increase or decrease this batch size depending on your
        # memory requirements of your computer / model
        # This will not affect the performance of the model and your score
        batch_size = 64
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    # Ensure your model outputs values in the [0, 255] range before this step! If you normalized your data to [0, 1], you must multiply by 255 before saving.
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # You don't need to change the code below
    # Load the data
    train_data_input, train_data_label, test_data_input = load_data()
    # Train the model
    model = training(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    # The name of the submission file is submit_this_test_data_output.npz
    testing(model, test_data_input)

    return None


if __name__ == "__main__":
    main()
