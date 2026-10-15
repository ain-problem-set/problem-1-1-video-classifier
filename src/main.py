"""CLI entry point: python src/main.py <command>."""
import argparse
import sys

# Each entry: name -> (help, configure(parser), handler(args) -> int)
SUBCOMMANDS = {}


def _read(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def cmd_sample(args):
    import config as C
    from sampling import fetch_parquet, load_videomme, stratified_sample
    parquet = fetch_parquet(C.VIDEOMME_PARQUET_URL, C.DATA / "videomme.parquet")
    plan = stratified_sample(load_videomme(parquet))
    C.ART.mkdir(parents=True, exist_ok=True)
    plan.to_csv(C.ART / "sample_plan.csv", index=False)
    print(plan.groupby("role").size().to_string())


SUBCOMMANDS["sample"] = ("draw the stratified 10% VideoMME sample", lambda p: None, cmd_sample)


def cmd_download(args):
    import config as C
    from download import download_sample
    final, missing = download_sample(_read(C.ART / "sample_plan.csv"))
    final.to_csv(C.ART / "sample.csv", index=False)
    print(f"{len(final)} videos ready, {int((final.replaced != '').sum())} replaced by spares, "
          f"unfilled slots: {missing or 'none'}")


SUBCOMMANDS["download"] = ("download sampled videos (yt-dlp, <=240p)", lambda p: None, cmd_download)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="main.py", description="Video type classifier for chunking routing")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, (help_text, configure, handler) in SUBCOMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        configure(p)
        p.set_defaults(func=handler)
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
