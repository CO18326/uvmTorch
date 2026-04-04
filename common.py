"""
Common utilities for CUDA graph examples.

This module contains shared model definitions and training functions used
across different CUDA graph demonstration scripts.
"""

import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
#from datasets import load_dataset


class MLPBlock(nn.Module):
    """Single MLP block with Linear, ReLU, and Dropout."""

    def __init__(self, in_features, out_features, dropout_p=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MLPModel(nn.Module):
    """MLP model with three consecutive MLP blocks plus a final linear layer."""

    def __init__(self,
                 input_dim,
                 hidden_dim1,
                 hidden_dim2,
                 hidden_dim3,
                 output_dim,
                 dropout_p1=0.2,
                 dropout_p2=0.1,
                 dropout_p3=0.1):
        super().__init__()
        self.block1 = MLPBlock(input_dim, hidden_dim1, dropout_p=dropout_p1)
        self.block2 = MLPBlock(hidden_dim1, hidden_dim2, dropout_p=dropout_p2)
        self.block3 = MLPBlock(hidden_dim2, hidden_dim3, dropout_p=dropout_p3)
        self.output = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x


def train_without_cuda_graph(model,
                             loss_fn,
                             optimizer,
                             data_loader,
                             profiler=None):
    """Train without using CUDA graph (standard PyTorch training)."""
    print("Training WITHOUT CUDA graph...")

    for i, data in enumerate(data_loader):
        #with record_function("## optimizer.zero_grad ##"):
        optimizer.zero_grad()

        #with record_function("## forward_pass ##"):
        y_pred = model(data["input_ids"].to("cuda"),labels=data["labels"].to("cuda"))

        #with record_function("## loss_computation ##"):
        loss = y_pred.loss

        #with record_function("## backward_pass ##"):
        loss.backward()

        #with record_function("## optimizer.step ##"):
        optimizer.step()

        if profiler is not None:
            profiler.step()

        print(i)
        
        if i==4:
            break
        

        # NOTE: Avoid calling .item() in the training loop as it triggers device-to-host
        # memory copy and CPU-GPU synchronization, which damages performance.
        # if i % 2 == 0:
        #     print(f"  Iteration {i+1:2d}: Loss = {loss.item():.4f}")

    print(f"  Completed iterations.")
    print()


def setup_model_and_data(device):
    """Setup model configuration and generate training data."""
    # Model setup
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-base")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    
    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=2000,
        )
        labels = []
        for seq in model_inputs["input_ids"]:
            labels.append([tok if tok != tokenizer.pad_token_id else -100 for tok in seq])
        model_inputs["labels"] = labels
        return model_inputs

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    train_loader = DataLoader(dataset,batch_size=4,shuffle=True)
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return train_loader


def create_model(config, device):
    """Create a new MLPModel instance with the given configuration."""
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-base",
        torch_dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    return model


def create_profiler():
    """Create a profiler with standard configuration."""
    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=schedule(wait=1, warmup=2, active=7, repeat=1),
                   record_shapes=True,
                   profile_memory=True,
                   with_stack=True)


def save_and_print_profile(prof, trace_file, scenario_name):
    """Save profiling trace and print summary."""
    # Create directory if it doesn't exist
    trace_dir = os.path.dirname(trace_file)
    if trace_dir and not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    prof.export_chrome_trace(trace_file)
    print(f"Profiling trace saved to: {trace_file}")
    print()

    print(f"Top 10 operations by CUDA time ({scenario_name}):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()
