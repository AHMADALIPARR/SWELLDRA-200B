import argparse
from swelldra.modeling.model import SwelldraMoE
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/moe_1m.yaml")
    parser.add_argument("--mode", choices=["train","infer","benchmark","test"], default="test")
    args = parser.parse_args()
    model = SwelldraMoE.from_config_file(args.config)
    params = model.count_parameters()
    print(f"Loaded {args.config}: Total {params['total']:,} Active {params['active_per_token']:,}")
    if args.mode in ["test","benchmark"]:
        from benchmarks.run_all import run_benchmarks
        run_benchmarks()
if __name__ == "__main__":
    main()
