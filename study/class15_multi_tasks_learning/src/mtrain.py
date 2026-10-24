"""Train one six-class model on data1 and data2, without using data3."""

from train import main as training_main


def main(argv=None):
    return training_main(argv, combined=True)


if __name__ == "__main__":
    raise SystemExit(main())
