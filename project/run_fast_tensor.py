import argparse
import sys
import time

import minitorch
import minitorch.datasets as datasets


def RParam(*shape, backend):
    r = 2 * (minitorch.rand(shape, backend=backend) - 0.5)
    r = r.detach()
    r.requires_grad_(True)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        self.bias = RParam(out_size, backend=backend)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(self.out_size)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        super().__init__()
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, hidden, backend)
        self.layer3 = Linear(hidden, 1, backend)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


def train(dataset_name, hidden_size=100, epochs=500, learning_rate=0.05, backend_name="cpu"):
    # Select backend
    if backend_name == "cpu":
        from minitorch.fast_ops import FastOps
        backend = minitorch.TensorBackend(FastOps)
    elif backend_name == "gpu":
        from minitorch.cuda_ops import CudaOps
        backend = minitorch.TensorBackend(CudaOps)
    else:
        from minitorch.tensor_ops import SimpleOps
        backend = minitorch.TensorBackend(SimpleOps)

    # Load dataset
    dataset_fn = getattr(datasets, dataset_name)
    PTS = 50
    points, labels = dataset_fn(PTS)
    N = len(points)

    # Create model
    model = Network(hidden_size, backend)
    optim = minitorch.SGD(model.parameters(), learning_rate)

    # Convert data to tensors with the chosen backend
    X = minitorch.tensor(points, backend=backend)
    y = minitorch.tensor(labels, backend=backend)

    times = []
    for epoch in range(1, epochs + 1):
        start = time.time()

        # Zero grads
        optim.zero_grad()

        # Forward
        out = model.forward(X).view(N)
        prob = (out * y) + (out - 1.0) * (y - 1.0)
        loss = -prob.log()
        (loss / N).sum().view(1).backward()
        total_loss = loss.sum().view(1)[0]

        # Update
        optim.step()

        epoch_time = time.time() - start
        times.append(epoch_time)

        # Logging
        if epoch % 10 == 0 or epoch == epochs:
            y2 = minitorch.tensor(labels, backend=backend)
            correct = int(((out.detach() > 0.5) == y2).sum()[0])
            print(
                f"Epoch {epoch:>3d} | Loss {total_loss:>8.4f} | "
                f"Correct {correct}/{N} | Time {epoch_time:.4f}s"
            )

    avg_time = sum(times[-50:]) / len(times[-50:])
    print(f"\nAvg time per epoch (last 50): {avg_time:.4f}s")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Tensor Training Benchmark")
    parser.add_argument("--BACKEND", type=str, default="cpu", choices=["cpu", "gpu", "simple"])
    parser.add_argument("--HIDDEN", type=int, default=100)
    parser.add_argument("--DATASET", type=str, default="simple", choices=["simple", "diag", "split", "xor"])
    parser.add_argument("--EPOCHS", type=int, default=500)
    parser.add_argument("--RATE", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Training on {args.DATASET} with {args.BACKEND} backend, hidden={args.HIDDEN}")
    import traceback
    try:
        train(args.DATASET, args.HIDDEN, args.EPOCHS, args.RATE, args.BACKEND)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
