import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import click
import wandb
from torch.utils.data import DataLoader
from rich.console import Console
from rich.text import Text
import numpy as np

from model import CNN, RNN, MLP
from utils import build_word2idx, build_embedding, load_word2vec, SentimentDataset
from train import train_model
from evaluate import evaluate_model

console = Console()

CONFIG = {
    "data_dir": "data",
    "train_file": "train.txt",
    "valid_file": "validation.txt",
    "test_file": "test.txt",
    "embedding_file": "wiki_word2vec_50.bin", 
    "results_dir": "results",
    "model_save_dir": "results/saved_models",

    "model_type": "cnn",
    "embedding_dim": 50,
    "max_len": 128,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 10,
    "dropout": 0.5,
    "early_stopping_patience": 3,
    "freeze_embeddings": True,

    "cnn_num_filters": 100,
    "cnn_filter_sizes": [3, 4, 5],

    "rnn_hidden_dim": 128,
    "rnn_num_layers": 2,
    "rnn_bidirectional": True,
    "rnn_type": "GRU",

    "mlp_hidden_dim": 64,

    "wandb_project": "Intro-to-AI",
    "wandb_notes": "assignment 2",
    "seed": 42,
}

@click.command()
@click.option('--model_type', default=CONFIG['model_type'], type=click.Choice(['cnn', 'rnn', 'mlp']), help='Model type to train.')
@click.option('--data_dir', default=CONFIG['data_dir'], type=click.Path(exists=True), help='Directory for data files.')
@click.option('--embedding_file', default=CONFIG['embedding_file'], type=str, help='Name of the embedding file in data_dir.')
@click.option('--num_epochs', default=CONFIG['num_epochs'], type=int, help='Number of training epochs.')
@click.option('--batch_size', default=CONFIG['batch_size'], type=int, help='Batch size.')
@click.option('--learning_rate', default=CONFIG['learning_rate'], type=float, help='Learning rate.')
@click.option('--dropout', default=CONFIG['dropout'], type=float, help='Dropout rate.')
@click.option('--embedding_dim', default=CONFIG['embedding_dim'], type=int, help='Embedding dimension (must match pretrained file).')
@click.option('--max_len', default=CONFIG['max_len'], type=int, help='Max sequence length for padding/truncation.')
@click.option('--freeze_embeddings', default=CONFIG['freeze_embeddings'], type=bool, help='Freeze embedding layer weights.')
@click.option('--early_stopping_patience', default=CONFIG['early_stopping_patience'], type=int, help='Patience for early stopping.')
@click.option('--seed', default=CONFIG['seed'], type=int, help='Random seed for reproducibility.')

@click.option('--cnn_num_filters', default=CONFIG['cnn_num_filters'], type=int, help='Number of CNN filters per size.')
@click.option('--cnn_filter_sizes', default=CONFIG['cnn_filter_sizes'], type=str, help='Comma-separated list of CNN filter sizes.')

@click.option('--rnn_hidden_dim', default=CONFIG['rnn_hidden_dim'], type=int, help='RNN hidden dimension.')
@click.option('--rnn_num_layers', default=CONFIG['rnn_num_layers'], type=int, help='Number of RNN layers.')
@click.option('--rnn_bidirectional', default=CONFIG['rnn_bidirectional'], type=bool, help='Use bidirectional RNN.')
@click.option('--rnn_type', default=CONFIG['rnn_type'], type=click.Choice(['GRU', 'LSTM']), help='Type of RNN unit.')

@click.option('--mlp_hidden_dim', default=CONFIG['mlp_hidden_dim'], type=int, help='MLP hidden dimension.')

@click.option('--wandb_project', default=CONFIG['wandb_project'], type=str, help='W&B project name.')
@click.option('--wandb_notes', default=CONFIG['wandb_notes'], type=str, help='Notes for W&B run.')
@click.option('--wandb_disabled', is_flag=True, help="Disable Weights & Biases logging.")
def main(**cli_config): 
    """Main script to train and evaluate sentiment analysis models."""

    config = {**CONFIG, **{k: v for k, v in cli_config.items() if v is not None}}
    if cli_config['cnn_filter_sizes'] is not None:
        config['cnn_filter_sizes'] = [int(size) for size in cli_config['cnn_filter_sizes'].split(',')]

    run = None  

    if config['wandb_disabled']:
        os.environ["WANDB_DISABLED"] = "true"
        console.print(Text("Weights & Biases logging is DISABLED.", style="bold yellow"))
    else:
        run_name = f"{config['model_type']}_{config.get('rnn_type','').lower()}_lr{config['learning_rate']}_bs{config['batch_size']}_{int(time.time())}"
        if config['model_type'] != 'rnn':
            run_name = f"{config['model_type']}_lr{config['learning_rate']}_bs{config['batch_size']}_{int(time.time())}"
        try:
            run = wandb.init(
                project=config['wandb_project'],
                entity=config.get('wandb_entity'),
                config=config,
                name=run_name,
                notes=config['wandb_notes'],
                reinit=True,
                save_code=True
            )
            if run:
                console.print(Text(f"W&B Run URL: {run.url}", style=f"link {run.url}"))
        except Exception as e:
            console.print(Text(f"Could not initialize W&B: {e}. Disabling W&B for this run.", style="bold red"))
            os.environ["WANDB_DISABLED"] = "true"

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"Using device: {device}")

    train_file_path = os.path.join(config['data_dir'], config['train_file'])
    valid_file_path = os.path.join(config['data_dir'], config['valid_file'])
    test_file_path = os.path.join(config['data_dir'], config['test_file'])
    embedding_file_path = os.path.join(config['data_dir'], config['embedding_file'])
    os.makedirs(config['model_save_dir'], exist_ok=True)

    console.print("Loading and preprocessing data...")
    word2idx= build_word2idx(train_file_path)
    vocab_size = len(word2idx)
    console.print(f"Vocabulary size: {vocab_size}")

    pretrained_embedding_matrix = load_word2vec(embedding_file_path)
    pretrained_embedding_matrix = build_embedding(word2idx, pretrained_embedding_matrix)
    
    if isinstance(pretrained_embedding_matrix, torch.Tensor):
        pretrained_embedding_matrix_np = pretrained_embedding_matrix.numpy()
    else:
        pretrained_embedding_matrix_np = pretrained_embedding_matrix


    console.print("Creating data loaders...")
    train_loader = DataLoader(SentimentDataset(train_file_path, word2idx, config['max_len']), config['batch_size'], shuffle=True)
    val_loader = DataLoader(SentimentDataset(valid_file_path, word2idx, config['max_len']), config['batch_size'], shuffle=False)
    test_loader = DataLoader(SentimentDataset(test_file_path, word2idx, config['max_len']), config['batch_size'], shuffle=False)

    output_dim = 1
    model = None
    model_name_str = config['model_type'].upper()

    console.print(f"Initializing {model_name_str} model...")
    if config['model_type'] == 'cnn':
        model = CNN(vocab_size=vocab_size,
                    embedding_dim=config['embedding_dim'],
                    num_filters=config['cnn_num_filters'],
                    filter_sizes=config['cnn_filter_sizes'],
                    output_dim=output_dim,
                    dropout=config['dropout'],
                    pretrained_embedding=pretrained_embedding_matrix_np,
                    freeze_embeddings=config['freeze_embeddings'])
    elif config['model_type'] == 'rnn':
        model_name_str = f"{config['rnn_type'].upper()}-RNN"
        model = RNN(vocab_size=vocab_size,
                    embedding_dim=config['embedding_dim'],
                    hidden_dim=config['rnn_hidden_dim'],
                    output_dim=output_dim,
                    num_layers=config['rnn_num_layers'],
                    bidirectional=config['rnn_bidirectional'],
                    dropout=config['dropout'],
                    pretrained_embedding=pretrained_embedding_matrix_np,
                    freeze_embeddings=config['freeze_embeddings'])
    elif config['model_type'] == 'mlp':
        model = MLP(vocab_size=vocab_size,
                    embedding_dim=config['embedding_dim'],
                    hidden_dim=config['mlp_hidden_dim'],
                    output_dim=output_dim,
                    dropout=config['dropout'],
                    pretrained_embedding=pretrained_embedding_matrix_np,
                    freeze_embeddings=config['freeze_embeddings'])
    else:
        console.print(Text(f"Error: Invalid model type '{config['model_type']}' specified.", style="bold red"))
        return

    console.print(f"Initialized {model_name_str} model.")
    if wandb.run:
        wandb.config.update({"model_architecture": model_name_str})

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss().to(device)

    model_filename = f"{config['model_type']}_best_model_lr{config['learning_rate']}.pt"
    model_save_path = os.path.join(config['model_save_dir'], model_filename)
    console.print(f"Best model will be saved to: {model_save_path}")

    best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config['num_epochs'],
        device=device,
        model_save_path=model_save_path,
        early_stopping_patience=config['early_stopping_patience'],
        config=config
    )

    console.print(Text("\nEvaluating best model on the test set...", style="bold green"))
    evaluate_model(best_model, test_loader, criterion, device, is_test=True, model_name=model_name_str)

    if wandb.run:
        run.finish()

if __name__ == '__main__':
    main()
