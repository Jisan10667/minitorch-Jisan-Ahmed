import minitorch
from minitorch import tensor, Tensor
from minitorch.nn import maxpool2d, logsoftmax, dropout
from minitorch import Conv2dFun
from minitorch.fast_ops import FastOps
from minitorch.tensor_ops import TensorBackend

FastBackend = TensorBackend(FastOps)


class CNN(minitorch.Module):
    def __init__(self):
        super().__init__()

        # Q1: First conv layer — takes grayscale input (1 channel),
        #     produces 4 feature maps, using a 3x3 kernel.
        self.conv1 = minitorch.Conv2d(1, 4, (3, 3))

        # Q2: Second conv layer — takes the 4 feature maps from conv1,
        #     produces 8 feature maps, using a 3x3 kernel.
        self.conv2 = minitorch.Conv2d(4, 8, (3, 3))

        # Q3: After two 2x2 max-pools, spatial dimensions go 28 → 14 → 7.
        #     With 8 channels and 7x7 spatial size, the flattened size is:
        self.fc1 = minitorch.Linear(392, 64)
        self.fc2 = minitorch.Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, 1, 28, 28)

        # Conv block 1: convolve, activate, then pool
        x = self.conv1(x).relu()          # Q4: activation function
        x = maxpool2d(x, (2, 2))  # Q5: pool kernel size

        # Conv block 2: same pattern
        x = self.conv2(x).relu()          # Q6: activation function
        x = maxpool2d(x, (2, 2))  # Q7: pool kernel size

        # Flatten: collapse spatial dims into a single vector per image
        batch = x.shape[0]
        x = x.view(batch, 392)           # Q8: flattened size

        # Fully connected block
        x = self.fc1(x).relu()            # Q9: activation function
        x = dropout(x, 0.5, ignore=not self.training)  # Q10: dropout rate

        x = self.fc2(x)

        # Q11: Convert raw scores to log-probabilities.
        #      Which dimension represents the 10 classes?
        return logsoftmax(x, dim=1)


def train_mnist():
    from sklearn.datasets import fetch_openml
    import numpy as np

    # Load MNIST — 70,000 images of handwritten digits (28x28 pixels)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images, labels = mnist.data, mnist.target.astype(int)

    # Shuffle and split into train/test
    n_train, n_test = 5000, 500
    idx = np.random.permutation(len(images))
    train_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_test]

    def img_to_list(flat_img):
        """Convert a flat 784-pixel array into a [28, 28] nested list, normalized to [0, 1]."""
        return [[(float(flat_img[i * 28 + j]) / 255.0) for j in range(28)] for i in range(28)]  # Q12

    # Training data kept as Python lists (we'll build tensors per batch)
    X_train_list = [[img_to_list(images[i])] for i in train_idx]
    y_train_list = [float(labels[i]) for i in train_idx]

    # Test data as a single tensor (evaluated in one pass)
    X_test = minitorch.tensor([
        [img_to_list(images[i])] for i in test_idx
    ], backend=FastBackend)
    y_test = minitorch.tensor([float(labels[i]) for i in test_idx], backend=FastBackend)

    model = CNN()
    optimizer = minitorch.SGD(model.parameters(), lr=0.5)  # Q13, Q14

    import random
    BATCH_SIZE = 32  # Q15
    n_train = len(X_train_list)

    for epoch in range(20):
        model.train()  # Q16: Set model to training mode

        indices = list(range(n_train))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            batch_idx = indices[start:end]
            bs = end - start

            # Build batch tensors from the Python lists
            X_batch = minitorch.tensor(
                [X_train_list[i] for i in batch_idx],
                backend=FastBackend
            )
            y_batch = minitorch.tensor(
                [y_train_list[i] for i in batch_idx],
                backend=FastBackend
            )

            # Q17: Reset gradients from the previous batch
            optimizer.zero_grad()

            # Forward pass: get log-probabilities
            log_probs = model.forward(X_batch)

            # Q18: Compute NLL loss.
            # one_hot converts labels [3, 7, 1] into a matrix with 1.0 at the correct class.
            # Multiplying by log_probs picks out log P(correct class) for each sample.
            # We negate and average because we want to MINIMIZE negative log-likelihood.
            loss = -(log_probs * minitorch.one_hot(y_batch, 10)).sum() / bs

            # Q19: Compute gradients via backpropagation
            loss.backward()

            # Q20: Update weights using the computed gradients
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches


        # Switch to evaluation mode
        model.eval()  # Q21

        test_probs = model.forward(X_test)

        # Q22: Get the predicted class for each test image.
        # argmax returns a one-hot tensor, so we use it with one_hot targets.
        predictions = minitorch.argmax(test_probs, dim=1)
        targets = minitorch.one_hot(y_test, 10)  # Q23

        # Accuracy: predictions and targets are both one-hot.
        # Their element-wise product is 1 only where both agree.
        accuracy = (predictions * targets).sum() / y_test.size

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy.item():.2%}")


if __name__ == "__main__":
    train_mnist()
