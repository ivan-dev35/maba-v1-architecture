import argparse
import sys
import torch

from .config import Config
from .model import Model
from .tokenizer import Tokenizer
from .generate import spec_gen
from .train import train
from .export_weights import export_bin
from .hardware import get_device

def main():
    parser = argparse.ArgumentParser(description="Maba v1 Architecture CLI")
    sub = parser.add_subparsers(dest="command")

    # Generate
    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--prompt", type=str, default="def fib(n):", help="Input prompt")
    p_gen.add_argument("--max-tokens", type=int, default=32, help="Max new tokens")
    p_gen.add_argument("--weights", type=str, default=None, help="Path to checkpoint .pt")
    p_gen.add_argument("--speculative", action="store_true", help="Use MTP speculative decoding")
    p_gen.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/mps/auto)")

    # Train
    p_tr = sub.add_parser("train", help="Run training")
    p_tr.add_argument("--steps", type=int, default=30, help="Training steps")
    p_tr.add_argument("--batch-size", type=int, default=2, help="Batch size")
    p_tr.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    p_tr.add_argument("--device", type=str, default="auto", help="Device")
    p_tr.add_argument("--output", type=str, default="maba_checkpoint.pt", help="Output checkpoint path")

    # Export
    p_exp = sub.add_parser("export", help="Export weights to binary format for C++")
    p_exp.add_argument("--checkpoint", type=str, default=None, help="Input PyTorch checkpoint")
    p_exp.add_argument("--output", type=str, default="maba_weights.bin", help="Output binary file")

    # Params
    p_par = sub.add_parser("params", help="Audit model parameter topology")
    p_par.add_argument("--scale", type=str, default="100M", choices=["100M", "1B", "3B", "7B"], help="Model scale preset")

    args = parser.parse_args()

    if args.command == "generate":
        dev = get_device(args.device)
        cfg = Config()
        model = Model(cfg).to(dev)
        if args.weights:
            sd = torch.load(args.weights, map_location=dev, weights_only=True)
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            model.load_state_dict(sd)
        tok = Tokenizer()

        if args.speculative:
            spec_gen(model, tok, args.prompt, max_new_tokens=args.max_tokens, device=str(dev))
        else:
            ids = torch.tensor([tok.encode(args.prompt, add_bos=True)], device=dev)
            out = model.generate(ids, max_new_tokens=args.max_tokens)
            print(tok.decode(out[0].tolist()))

    elif args.command == "train":
        dev = str(get_device(args.device))
        train(n_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, ckpt_path=args.output, device=dev)

    elif args.command == "export":
        cfg = Config()
        model = Model(cfg)
        if args.checkpoint:
            sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            model.load_state_dict(sd)
        export_bin(model, args.output)

    elif args.command == "params":
        cfg = Config.from_preset(args.scale)
        counts = cfg.compute_param_count()
        print(f"Maba Architecture: {args.scale.upper()} Topology Audit")
        print("-" * 42)
        for k, v in counts.items():
            if isinstance(v, int):
                print(f"{k:<20}: {v:>16,}")
            else:
                print(f"{k:<20}: {v:>16}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
