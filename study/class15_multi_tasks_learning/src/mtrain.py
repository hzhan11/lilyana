"""Train six classes with 40 train + 10 validation images each from data1 and data2."""

from train import main as training_main


def main(argv=None):
    return training_main(argv, combined=True)


if __name__ == "__main__":
    raise SystemExit(main())
