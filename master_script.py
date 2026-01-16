import subprocess
import argparse

SEQ_LIST = [512,650,720,820]
BATCH_LIST = [8]

def run_once(python_script, fixed_args, batch, seq):
    """Run training script once with batch + seq."""
    cmd = [
        "python", python_script,
        "--batch_size", str(batch),
        "--seq_len", str(seq),
    ] + fixed_args

    print("\n====================================================")
    print(f"Running: batch_size={batch}, seq_len={seq}")
    print("CMD:", " ".join(cmd))
    print("====================================================\n")

    subprocess.run(cmd, check=False)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--python_script", type=str, required=True,
                        help="Your training script, e.g., train.py")

    
    parser.add_argument("fixed", nargs=argparse.REMAINDER,
                        help="All other args passed as-is to the Python script")

    args = parser.parse_args()

    
    fixed_args = []
    skip_keys = {"--batch_size", "--seq_len"}

    i = 0
    while i < len(args.fixed):
        if args.fixed[i] in skip_keys:
            
            i += 2
            continue
        fixed_args.append("--"+args.fixed[i])
        i += 1

    
    for batch in BATCH_LIST:
        for seq in SEQ_LIST:
            run_once(args.python_script, fixed_args, batch, seq)

if __name__ == "__main__":
    main()
