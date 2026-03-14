import argparse
import json
import sys
import types
from pathlib import Path

import torch


def _ensure_monai_trunc_normal() -> None:
    """Provide a minimal monai.networks.layers.trunc_normal_ fallback when MONAI is unavailable."""
    try:
        from monai.networks.layers import trunc_normal_  # noqa: F401
        return
    except Exception:
        pass

    from torch.nn.init import trunc_normal_

    monai_mod = types.ModuleType("monai")
    networks_mod = types.ModuleType("monai.networks")
    layers_mod = types.ModuleType("monai.networks.layers")
    layers_mod.trunc_normal_ = trunc_normal_

    monai_mod.networks = networks_mod
    networks_mod.layers = layers_mod

    sys.modules.setdefault("monai", monai_mod)
    sys.modules.setdefault("monai.networks", networks_mod)
    sys.modules.setdefault("monai.networks.layers", layers_mod)


def build_model() -> torch.nn.Module:
    _ensure_monai_trunc_normal()
    from WaveFFTVNet.networks.model1 import VNet

    return VNet(n_channels=1, n_classes=14, n_filters=16, has_residual=False)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            _ = model(input_tensor)

    total_flops = 0.0
    for evt in prof.key_averages():
        total_flops += float(getattr(evt, "flops", 0) or 0)
    return total_flops


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Params (M) and FLOPs (G) for model1.py VNet.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--depth", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-path", type=Path, default=Path("complexity_result.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    x = torch.randn(args.batch_size, args.in_channels, args.depth, args.height, args.width, device=device)

    params = count_params(model)
    flops = estimate_flops(model, x)

    mparams = round(params / 1e6, 2)
    gflops = round(flops / 1e9, 2)

    results = {
        "input_shape": [args.batch_size, args.in_channels, args.depth, args.height, args.width],
        "device": str(device),
        "params": params,
        "flops": int(flops),
        "MParams": mparams,
        "GFLOPs": gflops,
    }

    args.save_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"MParams: {mparams:.2f}")
    print(f"GFLOPs: {gflops:.2f}")
    print(f"Saved to: {args.save_path}")


if __name__ == "__main__":
    main()
