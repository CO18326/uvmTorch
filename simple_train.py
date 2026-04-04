import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from torch.optim import AdamW
import time, argparse, statistics
import torch.cuda.nvtx as nvtx
from cpu_adm import DeepSpeedCPUAdam as ds_opt
import threading
import csv
from datetime import datetime
# ---- DeepSpeed Config ----

try:
    import pynvml
    pynvml.nvmlInit()
    _use_nvml = True
except ImportError:
    _use_nvml = False
    print("⚠️ pynvml not installed; GPU utilization logging disabled.")


def gpu_monitor(stop_event, device_id=0, interval=0.5, log_file="gpu_util_log.csv"):
    """Background thread to log GPU utilization periodically."""
    if not _use_nvml:
        return

    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gpu_util_pct", "gpu_mem_used_MB", "gpu_mem_total_MB"])

        while not stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                util.gpu,
                mem.used / (1024 ** 2),
                mem.total / (1024 ** 2)
            ])
            f.flush()
            print(f"[GPU MON] Util={util.gpu:3d}% | Mem={mem.used / (1024 ** 2):6.1f} MB")
            time.sleep(interval)


def clip_grad_norm_fp32(params, max_norm: float):
    """
    Gradient clipping in FP32 like DeepSpeed.
    Works on GPU tensors.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    # Compute total L2 norm (in FP32)
    total_norm = torch.norm(
        torch.stack([g.detach().float().norm(2) for g in grads]),
        2
    ).to(grads[0].device)

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # Scale all gradients if needed
    if clip_coef_clamped < 1.0:
        for g in grads:
            g.data.mul_(clip_coef_clamped)

    return total_norm


import psutil
import time
import threading
import os
import csv

'''def log_cpu_memory(interval=1.0, log_file="cpu_mem_log.csv", stop_event=None):
    """
    Periodically log CPU memory usage (RSS) of the current process.
    
    Args:
        interval (float): sampling interval in seconds
        log_file (str): CSV file to save readings
        stop_event (threading.Event): optional stop signal
    """
    pid = os.getpid()
    process = psutil.Process(pid)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_sec", "cpu_total_pct","rss_mb", "vms_mb"])
        start_time = time.time()

        while not (stop_event and stop_event.is_set()):
            mem_info = process.memory_info()
            cpu_total = psutil.cpu_percent(interval=None)
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
            ts = time.time() - start_time

            writer.writerow([f"{ts:.3f}",f"{cpu_total:.2f}", f"{rss_mb:.2f}", f"{vms_mb:.2f}"])
            f.flush()
            time.sleep(interval)'''




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=910)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="ibm-granite/granite-3.0-2b-base") #default="EleutherAI/gpt-neo-1.3B")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Needs GPU!"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name,torch_dtype=torch.bfloat16).cuda()
    model.train()

    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.seq_len,
        )
        labels = []
        for seq in model_inputs["input_ids"]:
            labels.append([tok if tok != tokenizer.pad_token_id else -100 for tok in seq])
        model_inputs["labels"] = labels
        return model_inputs

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)

    optimizer = ds_opt(model.parameters(), lr=5e-5)

    times = []
    '''stop_event = threading.Event()
    global _use_nvml
    if _use_nvml:
        monitor_thread = threading.Thread(
            target=gpu_monitor, args=(stop_event, 0, 0.5, "gpu_util_log_uvm_2048_offload_nhi.csv"), daemon=True
        )
        monitor_thread.start()'''
    
    for step, batch in enumerate(dataloader):
        if step >= args.steps:
            break
        
        start = time.time()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
        ):
            nvtx.range_push(f"Step {step} - Forward")
            outputs = model(input_ids, labels=labels)
            nvtx.range_pop()
            loss = outputs.loss
        nvtx.range_push(f"Step {step} - Backward")
        loss.backward()
        nvtx.range_pop()
        nvtx.range_push(f"Step {step} - Optimizer")

        #norm = clip_grad_norm_fp32(model.parameters(), max_norm=1.0)
        
        if True:
            first_param_name, first_param = next(model.named_parameters())
            print(f"Before step: {first_param_name} -> {first_param.flatten()[0].item()}")
        optimizer.step()
        nvtx.range_pop()
        optimizer.zero_grad()

        if True:
            updated_param_name, updated_param = next(model.named_parameters())
            print(f"After step:  {updated_param_name} -> {updated_param.flatten()[0].item()}")
        torch.cuda.synchronize()
        end = time.time()

        step_time = end - start
        print(step_time)
        #print(loss.item())
        times.append(step_time)

    '''stop_event.set()
    if _use_nvml:
        monitor_thread.join()
        pynvml.nvmlShutdown()'''
    
    avg_time = statistics.mean(times) if times else float("nan")
    print(f"SEQ_LEN={args.seq_len}, AVG_STEP_TIME={avg_time:.4f}")


if __name__ == "__main__":
    '''stop_event = threading.Event()
    mem_thread = threading.Thread(
    target=log_cpu_memory,
    kwargs={"interval": 1.0, "log_file": "cpu_mem_usage_2048_adam_cpu.csv", "stop_event": stop_event},
    daemon=True
)'''
    #mem_thread.start()
    
    torch._C._cuda_beginUvmAllocate()
    main()
    torch._C._cuda_endUvmAllocate()
    

    #stop_event.set()
    #mem_thread.join()
    #print("✅ CPU memory log written to cpu_mem_usage_2048_adam_cpu.csv")
