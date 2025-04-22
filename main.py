import click
import sys
import os
import torch
from model_inversion_attack import attack_dataset

@click.group()
def cli():
    pass

@cli.group()
def model_inversion():
    pass

@model_inversion.command(help='Attack images in dataset')
@click.option('--dataset_path', default='dataset/data_pgm', help='Path to the dataset directory')
@click.option('--output_dir', default='attack_results', help='Directory to save attack results')
@click.option('--iterations', default=50, help='Number of iterations in attack')
@click.option('--loss_function', default="crossEntropy", type=click.Choice(['crossEntropy', 'softmax']),
              help='Loss function to use: crossEntropy or softmax')
def attack_dataset_cmd(dataset_path, output_dir, iterations, loss_function):
    print(f"Attacking images in {dataset_path}")
    attack_dataset(dataset_path, output_dir, iterations, loss_function)

if __name__ == "__main__":
    cli()