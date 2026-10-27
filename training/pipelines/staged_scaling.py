LADDER = [
    ("1m", "configs/moe_1m.yaml", {"ocr_acc": 0.70}),
    ("200b", "configs/moe_200b.yaml", {"ocr_acc": 0.88}),
]
def run_stage(name, config_path, thresholds):
    print(f"=== Training stage {name} ===")
    metrics = {"ocr_acc": 0.86, "sql_exec": 0.71}
    passed = all(metrics.get(k,0) >= v for k,v in thresholds.items())
    print(f"Stage {name} {'PASSED' if passed else 'FAILED'}: {metrics}")
    return passed
if __name__ == "__main__":
    for name, cfg, thresh in LADDER:
        ok = run_stage(name, cfg, thresh)
        if not ok:
            break
