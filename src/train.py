import torch
import time
import os
import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
import wandb

from evaluate import evaluate_model 

console = Console()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=console.print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
            trace_func (function): Function to use for printing messages. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(Text(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...', style="green"))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_one_epoch(model, iterator, optimizer, criterion, device, epoch_num, total_epochs):
    """Trains the model for one epoch."""
    epoch_loss = 0
    epoch_correct = 0
    total_samples = 0

    model.train()

    with Progress(
        TextColumn(f"[bold blue]Epoch {epoch_num+1}/{total_epochs}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("• Loss: {task.fields[loss]:.4f}"),
        TimeElapsedColumn(),
        TextColumn("• ETA:"),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Training", total=len(iterator), loss=0.0)

        for batch in iterator:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()

            predictions = model(sequences).squeeze(1)

            loss = criterion(predictions, labels.float())

            preds_binary = torch.round(torch.sigmoid(predictions))
            correct = (preds_binary == labels).sum().item()
            epoch_correct += correct
            total_samples += labels.size(0)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress.update(task, advance=1, loss=loss.item())

    avg_epoch_loss = epoch_loss / len(iterator)
    avg_epoch_acc = epoch_correct / total_samples if total_samples > 0 else 0
    return avg_epoch_loss, avg_epoch_acc


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device,
                model_save_path, early_stopping_patience=5, config=None):
    """
    Orchestrates the training and validation process.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        num_epochs (int): Total number of epochs to train for.
        device (torch.device): The device to train on (CPU or CUDA).
        model_save_path (str): Path to save the best model checkpoint.
        early_stopping_patience (int): Patience for early stopping.
        config (dict, optional): Dictionary of hyperparameters passed from main.py, used for wandb.

    Returns:
        nn.Module: The best performing model loaded from the checkpoint.
    """
    console.print(Text(f"Training on {device}...", style="bold cyan"))
    model.to(device)

    if wandb.run and config.get("wandb_watch_model", True):
        wandb.watch(model, criterion, log=config.get("wandb_watch_log_type", "all"),
                    log_freq=config.get("wandb_watch_log_freq", 100))

    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=model_save_path)

    best_val_loss = float('inf')

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, device, is_test=False, model_name=config.get("model_type", "Model").upper()
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_end_time = time.time()
        epoch_mins, epoch_secs = divmod(epoch_end_time - epoch_start_time, 60)

        epoch_summary_table = Table(title=f"Epoch {epoch+1:02} Summary", show_header=True, header_style="bold magenta")
        epoch_summary_table.add_column("Metric", style="dim", width=20)
        epoch_summary_table.add_column("Value")

        epoch_summary_table.add_row("Epoch Time", f"{int(epoch_mins)}m {int(epoch_secs)}s")
        epoch_summary_table.add_row(Text("Train Loss", style="blue"), f"{train_loss:.4f}")
        epoch_summary_table.add_row(Text("Train Acc", style="blue"), f"{train_acc*100:.2f}%")
        epoch_summary_table.add_row(Text("Val. Loss", style="green"), f"{val_loss:.4f}")
        epoch_summary_table.add_row(Text("Val. Acc", style="green"), f"{val_acc*100:.2f}%")
        epoch_summary_table.add_row(Text("Val. F1", style="green"), f"{val_f1:.3f}")
        epoch_summary_table.add_row(Text("Val. Precision", style="green"), f"{val_precision:.3f}")
        epoch_summary_table.add_row(Text("Val. Recall", style="green"), f"{val_recall:.3f}")

        console.print(epoch_summary_table)

        if wandb.run:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss_epoch": train_loss,
                "train_acc_epoch": train_acc,
                "val_loss_epoch": val_loss,
                "val_acc_epoch": val_acc,
                "val_f1_epoch": val_f1,
                "val_precision_epoch": val_precision,
                "val_recall_epoch": val_recall,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_duration_secs": epoch_end_time - epoch_start_time
            }
            wandb.log(log_dict, step=epoch + 1)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            console.print(Text("Early stopping triggered.", style="bold red"))
            if wandb.run:
                wandb.log({"early_stopped_epoch": epoch + 1})
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    total_end_time = time.time()
    total_mins, total_secs = divmod(total_end_time - total_start_time, 60)
    console.print(Text(f'\nTotal Training Time: {int(total_mins)}m {int(total_secs)}s', style="bold cyan"))

    console.print(Text(f"Loading best model state from: {model_save_path}", style="yellow"))
    try:
        model.load_state_dict(torch.load(model_save_path))
        best_val_loss, best_val_acc, best_val_precision, best_val_recall, best_val_f1 = evaluate_model(
            model, val_loader, criterion, device, is_test=False, model_name=config.get("model_type", "Model").upper() + " (Best)"
        )
        if wandb.run:
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.summary["best_val_accuracy"] = best_val_acc
            wandb.summary["best_val_f1"] = best_val_f1
            wandb.summary["best_val_precision"] = best_val_precision
            wandb.summary["best_val_recall"] = best_val_recall
            console.print(Text(f"Logged best validation metrics to W&B Summary: Loss={best_val_loss:.4f}, Acc={best_val_acc*100:.2f}%", style="purple"))

    except FileNotFoundError:
        console.print(Text("Warning: Best model checkpoint not found. Cannot log best metrics to W&B summary or load for testing.", style="yellow"))
    except Exception as e:
         console.print(Text(f"Warning: Error loading best model state: {e}. Returning the last state.", style="yellow"))

    return model
