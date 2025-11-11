import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.prompt_vit import PromptTunedViT
from src.datasets.polyp_dataset import PolypDataset


def train_model(epochs=5, batch_size=8, lr=1e-4, save_path="experiments/checkpoints/prompt_vit_real.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training started on [{device}]...\n")

    model = PromptTunedViT(num_classes=2).to(device)
    dataset = PolypDataset(root_dir="data/train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"✅ Epoch {epoch+1} | Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Model saved at {save_path}")


if __name__ == "__main__":
    train_model()
