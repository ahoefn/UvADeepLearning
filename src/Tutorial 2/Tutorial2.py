import matplotlib
from matplotlib.colors import to_rgba
from tqdm import tqdm
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class SimpleClassifier(nn.Module):
        def __init__(self, num_inputs, num_hidden, num_outputs) -> None:
            super().__init__()

            # Define network:
            self.linear1 = nn.Linear(num_inputs, num_hidden)
            self.act_fn = nn.Tanh()
            self.linear2 = nn.Linear(num_hidden, num_outputs)

        def forward(self, x):
            x = self.linear1(x)
            x = self.act_fn(x)
            x = self.linear2(x)
            return x

    class XORDataset(Data.Dataset):
        def __init__(self, size, std=0.1) -> None:
            super().__init__()
            self.size = size
            self.std = std
            self.generate_continuous_xor()

        def generate_continuous_xor(self) -> None:
            data = torch.randint(
                low=0, high=2, size=(self.size, 2), dtype=torch.float32
            )
            labels = (data.sum(dim=1) == 1).to(torch.long)

            data += self.std * torch.randn(data.shape)

            self.data = data
            self.labels = labels

        def __len__(self):
            return self.size

        def __getitem__(self, id):
            data_point = self.data[id]
            data_label = self.labels[id]
            return data_point, data_label

        def visualize(self):
            # Separate data with respect to their labels:
            data = self.data.cpu().numpy()
            labels = self.labels.cpu().numpy()
            data_0 = data[labels == 0]
            data_1 = data[labels == 1]

            # Plotting
            plt.figure(figsize=(4, 4))
            plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0 ")
            plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
            plt.show()

    def train_model(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: Data.DataLoader,
        loss_module,
        num_epochs=100,
    ):
        # set model to train mode:
        model.train()

        # train
        for epoch in tqdm(range(num_epochs)):
            for data_inputs, data_labels in data_loader:
                # Get predictions:
                data_inputs: torch.Tensor = data_inputs.to(device)
                data_labels: torch.Tensor = data_labels.to(device)

                predictions: torch.Tensor = model(data_inputs)
                predictions = predictions.squeeze()

                # Get loss and step:
                loss = loss_module(predictions, data_labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def eval_model(model: nn.Module, data_loader: Data.DataLoader):
        model.eval()
        true_preds, num_preds = 0, 0
        with torch.no_grad():
            for data_inputs, data_labels in data_loader:
                data_inputs, data_labels = (
                    data_inputs.to(device),
                    data_labels.to(device),
                )
                predictions = model(data_inputs)
                predictions = predictions.squeeze(dim=1)
                predictions = torch.sigmoid(predictions)
                pred_labels = (predictions >= 0.5).long()

                true_preds += (pred_labels == data_labels).sum()
                num_preds += data_labels.shape[0]

        accuracy = true_preds / num_preds
        print(f"Accuracy obtained is: {accuracy}")

    @torch.no_grad()
    def visualize_classification(
        mode: nn.Module, data: torch.Tensor, labels: torch.Tensor
    ):
        npData = data.cpu().numpy()
        npLabels = labels.cpu().numpy()
        data_0 = npData[npLabels == 0]
        data_1 = npData[npLabels == 1]

        fig = plt.figure(figsize=(4, 4), dpi=500)
        plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0 ")
        plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")

        model.to(device)
        color0 = torch.Tensor(to_rgba("C0")).to(device)
        color1 = torch.Tensor(to_rgba("C1")).to(device)

        x1 = torch.arange(-0.5, 1.5, 0.01, device=device)
        x2 = torch.arange(-0.5, 1.5, 0.01, device=device)
        xx1, xx2 = torch.meshgrid(x1, x2, indexing="ij")

        model_inputs = torch.stack([xx1, xx2], dim=-1)
        predictions = model(model_inputs)
        predictions = torch.sigmoid(predictions)
        output_image = (1 - predictions) * color0[None, None] + predictions * color1[
            None, None
        ]
        output_image = output_image.cpu().numpy()
        plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
        plt.grid(False)
        return fig

    model = SimpleClassifier(2, 4, 1)
    model.to(device)
    loss_module = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_dataset = XORDataset(2500)
    train_data_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # print(model)
    train_model(model, optimizer, train_data_loader, loss_module)

    state_dict = model.state_dict()
    torch.save(state_dict, "src/model_states/test.tar")

    test_dataset = XORDataset(200)
    test_data_loader = Data.DataLoader(test_dataset, batch_size=8, shuffle=True)
    eval_model(model, test_data_loader)
    fig = visualize_classification(model, test_dataset.data, test_dataset.labels)
    plt.show()
