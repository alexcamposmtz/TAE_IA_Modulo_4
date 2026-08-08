import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms
    from torch.utils.tensorboard import SummaryWriter
    import math
    import marimo as mo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return SummaryWriter, datasets, device, mo, nn, plt, torch, transforms


@app.cell
def _(datasets, transforms):
    full_train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    len(full_train_dataset)
    return (full_train_dataset,)


@app.cell
def _(datasets, transforms):
    test_datset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    len(test_datset)
    return (test_datset,)


@app.cell
def _():
    id_label_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return (id_label_map,)


@app.cell
def _(id_label_map, plt):
    def plot_sample(data, label):
        plt.figure(figsize=(2, 2))
        plt.imshow(data.squeeze(), cmap="gray")
        plt.title(f"{id_label_map[label]}")
        plt.show()

    return


@app.cell
def _(full_train_dataset, torch):
    torch.manual_seed(42)

    N = 3000

    indices = torch.randperm(len(full_train_dataset))[:N]
    indices
    return (indices,)


@app.cell
def _(full_train_dataset, indices, torch):
    small_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    len(small_dataset)
    return (small_dataset,)


@app.cell
def _():
    # _idx = 1000
    # plot_sample(small_dataset[_idx][0], small_dataset[_idx][1])
    return


@app.cell
def _(small_dataset, torch):
    train_dataset, val_dataset = torch.utils.data.random_split(
        small_dataset,
        [2400, 600],
        generator=torch.Generator().manual_seed(42)
    )
    len(train_dataset), len(val_dataset)
    return train_dataset, val_dataset


@app.cell
def _(test_datset, torch, train_dataset, val_dataset):
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_datset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    len(train_loader), len(val_loader), len(test_loader)
    return test_loader, train_loader, val_loader


@app.cell
def _(nn):
    class MLP(nn.Module):
        def __init__(self, dropout_rate):
            super(MLP, self).__init__()
            self.dropout_rate = dropout_rate

            self.nn = nn.Sequential(
                nn.Flatten(),

                nn.Linear(28 * 28, 512), # Layer 1
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),

                nn.Linear(512, 256), # Layer 2
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),

                nn.Linear(256, 10) # Output layer
            )

        def forward(self, x):
            return self.nn(x)

    return (MLP,)


@app.cell
def _(mo):
    l2_lambda = mo.cli_args().get("l2_lambda") or 0.001
    patience = mo.cli_args().get("patience") or 10
    num_epochs = mo.cli_args().get("num_epochs") or 100
    dropout_rate = mo.cli_args().get("dropout_rate") or 0.2
    return dropout_rate, l2_lambda, num_epochs, patience


@app.cell
def _(MLP, device, dropout_rate, nn, torch):
    learning_rate = 0.01

    model = MLP(dropout_rate=dropout_rate).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()
    return loss_fn, model, optimizer


@app.cell
def _(SummaryWriter, dropout_rate, l2_lambda, num_epochs, patience):

    writer = SummaryWriter(f"runs/l2_reg_{l2_lambda}_patience_{patience}_epochs_{num_epochs}_dropout_{dropout_rate}")
    return (writer,)


@app.cell
def _(model):
    trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
    print(f"Number of trainable parameters: {trainable_params}")    
    return


@app.cell
def _(device, torch):
    @torch.no_grad()
    def evaluate(model, loss_fn, loader):
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            total_loss += loss.item()
            preds = torch.argmax(y_pred, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / len(loader), correct / total

    return (evaluate,)


@app.cell
def _(torch):
    def l2_regularization(model):
        l2_loss = 0.0

        for p in model.parameters():
            l2_loss += torch.sum(p**2)

        return l2_loss


    return (l2_regularization,)


@app.cell
def _(device, evaluate, l2_regularization, torch, writer):
    def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, l2_lambda, patience):
        epochs_no_improvement = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            model.train()

            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                y_pred = model(x)
                loss = loss_fn(y_pred, y) + l2_lambda * l2_regularization(model)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                preds = torch.argmax(y_pred, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total

            val_loss, val_accuracy = evaluate(model, loss_fn, val_loader)

            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_accuracy, "val": val_accuracy}, epoch)

            generalization_gap = abs(val_loss - train_loss)
            writer.add_scalar("generalization_gap", generalization_gap, epoch)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                f"Generalization Gap: {generalization_gap:.4f}, "
                f"Epochs without improvement: {epochs_no_improvement}/{patience}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                epochs_no_improvement = 0
                best_val_loss = val_loss
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                return  # Exit the training loop if early stopping is triggered
        


    return (train,)


@app.cell
def _(
    l2_lambda,
    loss_fn,
    model,
    num_epochs,
    optimizer,
    patience,
    train,
    train_loader,
    val_loader,
):
    train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs=num_epochs, l2_lambda=l2_lambda, patience=patience)
    return


@app.cell
def _(evaluate, loss_fn, model, test_loader, writer):
    test_loss, test_accuracy = evaluate(model, loss_fn, test_loader)

    writer.add_scalar("Loss/test", test_loss)
    writer.add_scalar("Accuracy/test", test_accuracy)
    writer.close()

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return


@app.cell
def _(model, torch, writer):
    torch.save(model.state_dict(), writer.log_dir + "/model.pth")
    return


if __name__ == "__main__":
    app.run()
