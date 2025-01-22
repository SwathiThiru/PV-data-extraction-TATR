import matplotlib.pyplot as plt
import torch

def plot_losses(model_checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

    # Retrieve losses
    training_losses = checkpoint.get('training_losses', [])
    validation_losses = checkpoint.get('validation_losses', [])
    #training_losses = [0.8, 0.6, 0.4, 0.2]
    #validation_losses = [0.9, 0.7, 0.5, 0.3]

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage

def main():
    plot_losses("../weights/new/structVisualization/with_modified_val_loss_4/model.pth")


if __name__ == "__main__":
    main()