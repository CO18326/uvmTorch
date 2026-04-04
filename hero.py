#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from cpu_adm import DeepSpeedCPUAdam as ds_opt
# -----------------------------
# Tensor address logger
# -----------------------------

class TensorAddressLogger(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}

        tensors = []

        def collect(x):
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, (list, tuple)):
                for i in x:
                    collect(i)
            elif isinstance(x, dict):
                for i in x.values():
                    collect(i)

        collect(args)
        collect(kwargs)

        info = []
        for t in tensors:
            info.append(
                f"addr=0x{t.data_ptr():x},shape={tuple(t.shape)},device={t.device}"
            )

        print(f"{func.__name__} | " + " | ".join(info))

        return func(*args, **kwargs)


# -----------------------------
# Dataset setup
# -----------------------------

def load_wikitext(device):

    batch = 4
    seq = 3000

    tokenizer = AutoTokenizer.from_pretrained(
        "ibm-granite/granite-3.3-2b-base"
    )

    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    text = "\n\n".join(dataset["text"][:200])

    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    inputs = []
    targets = []

    idx = 0

    while len(inputs) < 20:

        chunk = tokens[idx:idx+seq+1]

        if len(chunk) < seq+1:
            break

        x = chunk[:seq]
        y = chunk[1:seq+1]

        x = x.unsqueeze(0).repeat(batch,1)
        y = y.unsqueeze(0).repeat(batch,1)

        inputs.append(x.to(device))
        targets.append(y.to(device))

        idx += seq

    return inputs, targets, tokenizer.vocab_size


# -----------------------------
# CUDA graph preparation
# -----------------------------

def prepare_cuda_graph(model, loss_fn, optimizer, static_input, static_target):

    print("Warmup iterations...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):

            optimizer.zero_grad(set_to_none=True)

            outputs = model(static_input)
            logits = outputs.logits

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                static_target.view(-1)
            )

            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()

    print("Capturing CUDA graph...")

    g = torch.cuda.CUDAGraph()

    optimizer.zero_grad(set_to_none=True)

    with TensorAddressLogger():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.cuda.graph(g):

                outputs = model(static_input)
                logits = outputs.logits

                static_loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    static_target.view(-1)
                )

                static_loss.backward()
                optimizer.step()

    return g, static_loss


# -----------------------------
# Graph replay training
# -----------------------------

def train_with_cuda_graph(graph,
                          inputs,
                          targets,
                          static_input,
                          static_target):

    print("Running CUDA graph replay...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, (data, target) in enumerate(zip(inputs, targets)):

            static_input.copy_(data)
            static_target.copy_(target)

            with TensorAddressLogger():
                graph.replay()

            if i % 2 == 0:
                print("step", i)


# -----------------------------
# Main
# -----------------------------

def main():

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    # Load dataset
    inputs, targets, vocab = load_wikitext(device)

    batch = 4
    seq = 3000

    # Load Granite model
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-base",
        torch_dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = ds_opt(model.parameters(), lr=1e-5)
    #(model.parameters(), lr=1e-5)

    loss_fn = nn.CrossEntropyLoss()

    # Static tensors for CUDA graph
    static_input = torch.zeros(batch, seq, dtype=torch.long, device=device)
    static_target = torch.zeros(batch, seq, dtype=torch.long, device=device)

    # Prepare CUDA graph
    graph, static_loss = prepare_cuda_graph(
        model,
        loss_fn,
        optimizer,
        static_input,
        static_target
    )

    print("CUDA graph ready")

    # Train with graph replay
    train_with_cuda_graph(
        graph,
        inputs,
        targets,
        static_input,
        static_target
    )


if __name__ == "__main__":
    torch._C._cuda_beginUvmAllocate()
    main()
    torch._C._cuda_endUvmAllocate()
