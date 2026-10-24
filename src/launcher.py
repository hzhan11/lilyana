"""Dispatch lesson programs using the current Python interpreter."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = {
    "1": ("study/class1_dataset_prepare", {"app": "dataset_app.py"}),
    "2": ("study/class2_dataset_split", {"split": "split_dataset.py"}),
    "3": ("study/class3_training_model", {"training": "train_model.py"}),
    "4": ("study/class4_model_predict", {"app": "app.py", "predict": "inf_model.py"}),
    "5": ("study/class5_training_flow", {"training": "train_model.py"}),
    "6": ("study/class6_metrics", {
        "accuracy": "test_acc.py", "metrics": "test_metrics.py", "confusion": "test_tfpn.py",
    }),
    "7": ("study/class7_features_extraction", {
        "training": "train_model.py", "cnn": "train_cnn.py",
    }),
    "8": ("study/class8_cnn_playground", {"app": "app.py", "training": "app.py"}),
    "9": ("study/class9_overfitting_generalization", {
        "generate": "generate_data.py", "training": "train_model.py",
    }),
    "13": ("study/class13_house_price_regression", {
        "generate": "generate_data.py", "training": "train_model.py", "predict": "predict.py",
    }),
    "14": ("study/class14_lstm_forecasting", {
        "generate": "generate_data.py", "training": "train_model.py",
    }),
    "15": ("study/class15_multi_tasks_learning", {
        "app": "pic/app.py", "training": "train.py", "mtrain": "mtrain.py", "test": "test.py",
    }),
    "package1": ("package1", {
        "training": "train_breast_cancer.py", "figures": "make_teaching_images.py",
    }),
    "package2": ("package2", {
        "app": "demo_app.py", "training": "train_breast_cancer.py",
        "figures": "make_teaching_images.py",
    }),
}
UNIT_TEST_TARGETS = {"8", "9", "13", "14", "15", "package2"}


def source_dir(target):
    folder, _ = PROGRAMS[target]
    directory = ROOT.joinpath(*folder.split("/"))
    return directory if target.startswith("package") else directory / "src"


def resolve_target(value):
    for target, (folder, _) in PROGRAMS.items():
        if value in (target, f"class{target}", folder.rsplit("/", 1)[-1]):
            return target
    raise argparse.ArgumentTypeError(f"Unknown lesson: {value}. Use a launcher inside a class directory.")


def actions_for(target):
    actions = list(PROGRAMS[target][1])
    if target == "15":
        actions.append("check")
    elif target in UNIT_TEST_TARGETS or target == "6":
        actions.append("test")
    return actions


def commands_for(target, action):
    programs = PROGRAMS[target][1]
    if action == "install":
        raise ValueError(
            "Install dependencies once from the repository root: "
            "start.bat install (Windows) or sh start.sh install (Linux/macOS)."
        )
    if action == "check" and target == "15":
        return [["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"]]
    if action == "test" and target != "15":
        if target in UNIT_TEST_TARGETS:
            return [["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"]]
        if target == "6":
            return [[name] for name in programs.values()]
    action = "training" if action == "train" else action
    if action in programs:
        return [[programs[action]]]
    if action in programs.values():
        return [[action]]
    raise ValueError(
        f"Unknown action for {target}: {action}. Available: {', '.join(actions_for(target))}. "
        "You may also use an entry-point filename shown in readme.md."
    )


def run_action(target, action, args, parser):
    try:
        commands = commands_for(target, action)
    except ValueError as error:
        parser.error(str(error))
    directory = source_dir(target)
    if not directory.is_dir():
        parser.error(f"Source directory does not exist: {directory}")
    for command in commands:
        result = subprocess.run(
            [sys.executable, *command, *args], cwd=directory, check=False,
        )
        if result.returncode:
            return result.returncode
    return 0


def lesson_main(target, argv):
    folder, programs = PROGRAMS[target]
    details = ["Actions:"]
    details.extend(f"  {action:<11} {filename}" for action, filename in programs.items())
    if target == "15":
        details.append("              training 1 / training 2: train data1 / data2")
        details.append("              mtrain: train all six classes on data1 + data2")
        details.append("              test [1|2|m]: evaluate on pic/data3; omit model for 1 and 2")
        details.append("              test compare: compare all three models and save charts under runs")
        details.append("  check       Run this lesson's automated code tests")
    elif "test" in actions_for(target):
        details.append("  test        Run this lesson's tests/evaluation programs")
    details.extend([
        "",
        "train is an alias for training; entry-point filenames are also accepted.",
        "Install all dependencies once from the repository root:",
        "  start.bat install (Windows) / sh start.sh install (Linux/macOS)",
        "All class launchers automatically use the repository's shared .venv.",
        "Relative program paths are resolved from this lesson's src directory.",
    ])
    parser = argparse.ArgumentParser(
        prog="start.bat / sh start.sh",
        description=f"{folder.rsplit('/', 1)[-1]}: run this lesson's programs.",
        epilog="\n".join(details),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("action", nargs="?", help="Action or entry-point filename (omit to show help).")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed unchanged to the program.")
    options = parser.parse_args(argv)
    if options.action is None or options.action == "help":
        parser.print_help()
        return 0
    return run_action(target, options.action, options.args, parser)


def environment_python():
    environment = ROOT / ".venv"
    return environment / "Scripts" / "python.exe" if os.name == "nt" else environment / "bin" / "python"


def install_environment():
    requirements = ROOT / "src" / "requirements.txt"
    if not requirements.is_file():
        print(f"Shared requirements file does not exist: {requirements}", file=sys.stderr)
        return 1
    environment = ROOT / ".venv"
    python = environment_python()
    if environment.exists() and not python.is_file():
        print(
            f"Existing environment has no Python interpreter at {python}. "
            "Repair or move the incomplete .venv before running install again.",
            file=sys.stderr,
        )
        return 1
    if not python.is_file():
        print(f"Creating shared virtual environment: {environment}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(environment)], cwd=ROOT, check=False,
        )
        if result.returncode:
            return result.returncode
        if not python.is_file():
            print(f"Virtual environment creation did not produce {python}", file=sys.stderr)
            return 1
    print(f"Installing all lesson and package dependencies into {environment}", flush=True)
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(requirements)],
        cwd=ROOT, check=False,
    )
    if result.returncode:
        return result.returncode
    print("Setup complete. All class start scripts will use this shared .venv automatically.")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--lesson"]:
        lesson_parser = argparse.ArgumentParser(add_help=False)
        lesson_parser.add_argument("--lesson", required=True, type=resolve_target)
        lesson, remaining = lesson_parser.parse_known_args(argv)
        return lesson_main(lesson.lesson, remaining)
    parser = argparse.ArgumentParser(
        prog="start.bat / sh start.sh",
        description="Install all dependencies into one shared repository .venv.",
        epilog="Run install once here, then use start.bat / start.sh inside each class directory.",
    )
    parser.add_argument(
        "action", nargs="?", choices=("install", "help"),
        help="install: create/reuse .venv and install every lesson's dependencies; omit to show help.",
    )
    options = parser.parse_args(argv)
    if options.action == "install":
        return install_environment()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
