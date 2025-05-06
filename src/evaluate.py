import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rich.console import Console
from rich.table import Table
from rich.text import Text
import wandb

console = Console()

def evaluate_model(model, data_loader, criterion, device, is_test=False, model_name="Model"):
    """Evaluates the model on a given dataset (validation or test)."""
    epoch_loss = 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)

            predictions = model(sequences).squeeze(1)

            loss = criterion(predictions, labels.float())
            epoch_loss += loss.item()

            preds_binary = torch.round(torch.sigmoid(predictions))
            all_preds.extend(preds_binary.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = epoch_loss / len(data_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    if is_test:
        results_table = Table(title=f"{model_name} - Test Set Performance", show_header=True, header_style="bold blue")
        results_table.add_column("Metric", style="dim", width=20)
        results_table.add_column("Value")

        results_table.add_row("Test Loss", f"{avg_loss:.4f}")
        results_table.add_row("Accuracy", f"{accuracy*100:.2f}%")
        results_table.add_row("F1 Score", f"{f1:.3f}")
        results_table.add_row("Precision", f"{precision:.3f}")
        results_table.add_row("Recall", f"{recall:.3f}")
        console.print(results_table)

        if wandb.run:
            wandb.summary[f"test_loss_{model_name.lower().replace('-','_')}"] = avg_loss
            wandb.summary[f"test_accuracy_{model_name.lower().replace('-','_')}"] = accuracy
            wandb.summary[f"test_f1_{model_name.lower().replace('-','_')}"] = f1
            wandb.summary[f"test_precision_{model_name.lower().replace('-','_')}"] = precision
            wandb.summary[f"test_recall_{model_name.lower().replace('-','_')}"] = recall
            console.print(Text(f"Logged test metrics for {model_name} to W&B Summary.", style="purple"))

    return avg_loss, accuracy, precision, recall, f1 