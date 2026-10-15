"""CLI entry point: python src/main.py <command>."""
import argparse
import sys

# Each entry: name -> (help, configure(parser), handler(args) -> int)
SUBCOMMANDS = {}


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
