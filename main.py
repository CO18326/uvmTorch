import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
import deepspeed
import time
import torch.cuda.nvtx as nvtx

ds_config = {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "bf16": { "enabled": True },
    "zero_optimization": {
        "stage": 3,
        "offload_param": { 
        "device": "cpu", 
        "pin_memory": True },
        
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
        
    }
}
'''ds_config={
  "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "fp16": { "enabled": False },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    },
    "zenflow": {
      "topk_ratio": 0.05,
      "select_strategy": "auto",
      "select_interval": "auto",
      "update_interval": 4,
      "full_warm_up_rounds": 0,
      "overlap_step": True
    }
  }
}'''
'''ds_config={
    "train_batch_size": 2,
    "bf16": { "enabled": False },
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
      },
      "zenflow": {
            "topk_ratio": 0.01,
            "update_interval": 4,
            "full_warm_up_rounds": 0,
            "overlap_step": True
        }
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 2e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "gradient_accumulation_steps": 1,
    #"gradient_clipping": 1.0,
    "zero_allow_untested_optimizer": True
}'''

# ---- Load Model & Tokenizer ----
#model_name = "gpt2-large"            # 774M
#model_name = "gpt2-xl"               # 1.5B
#model_name = "EleutherAI/gpt-neo-1.3B"  # 1.3B
#model_name = "EleutherAI/gpt-neo-2.7B"  # 2.7B
#model_name="EleutherAI/gpt-neo-125M"
#model_name = "EleutherAI/gpt-neo-1.3B"
#model_name = "gpt2"
#model_name="EleutherAI/gpt-neo-125M"
model_name= "ibm-granite/granite-3.0-2b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)

optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(model.parameters(), lr=5e-5)

# Initialize DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# ---- Dataset ----
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=910)

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
dataloader = DataLoader(
    dataset,
    batch_size=ds_config["train_batch_size"],
    shuffle=True,
    collate_fn=default_data_collator
)

# ---- Training Loop ----
def train():
    print(model.named_parameters())
    step_times = []
    for epoch in range(1):

        for step, batch in enumerate(dataloader):
            start=time.time()
            inputs = batch["input_ids"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank) if "labels" in batch else inputs
            nvtx.range_push(f"Step {step} - Forward")
            outputs = model_engine(inputs, labels=labels)
            nvtx.range_pop()
            loss = outputs.loss
            nvtx.range_push(f"Step {step} - Backward")
            model_engine.backward(loss)
            #loss.backward()
            nvtx.range_pop()
            first_param_name, first_param = next(model.named_parameters())
            print(f"Before step: {first_param_name} -> {first_param.flatten()[0].item()}")
            nvtx.range_push(f"Step {step} - Optimizer")
            model_engine.step()

            #torch.cuda.empty_cache()
            nvtx.range_pop()

            #updated_param_name, updated_param = next(model.named_parameters())
            #print(f"After step:  {updated_param_name} -> {updated_param.flatten()[0].item()}")
            #torch.cuda.synchronize()
            step_time=time.time()-start
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f} Time {step_time}s")
            step_times.append(step_time)
            if step==20:
                break
    
    print("AVERAGE: ",sum(step_times)/len(step_times))


if __name__ == "__main__":
    #torch._C._cuda_beginUvmAllocate()
    train()
    #torch._C._cuda_endUvmAllocate()
