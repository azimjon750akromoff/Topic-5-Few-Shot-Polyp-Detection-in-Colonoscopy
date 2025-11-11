import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.prompt_vit import PromptTunedViT
from src.datasets.polyp_dataset import PolypDataset


def evaluate_model(model_path="experiments/checkpoints/prompt_vit_real.pt", batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Evaluation started on [{device}]...\n")

    model = PromptTunedViT(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = PolypDataset(root_dir="data/val")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("✅ Evaluation Results:")
    print(f"   Accuracy:  {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall:    {rec:.3f}")
    print(f"   F1-Score:  {f1:.3f}")

    return acc, prec, rec, f1


if __name__ == "__main__":
    evaluate_model()
