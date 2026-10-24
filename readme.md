# Lilyana: Machine Learning Lessons and Experiments

## Directory Layout

```text
lilyana/
  readme.md
  start.bat                   Windows shared-environment setup
  start.sh                    Linux / macOS / Git Bash shared-environment setup
  .venv/                      One shared environment for all lessons (Git-ignored)
  src/
    launcher.py               Shared launcher logic (Python standard library only)
    start.bat, start.sh       Internal Python interpreter helpers
    requirements.txt          Includes all lesson and package dependency lists
    test_launcher.py          Launcher tests
    main.py                   Original introductory Python example
  study/
    index.md                  Lesson index
    class1_dataset_prepare/
      readme.md               Lesson guide
      start.bat               Windows launcher for this lesson
      start.sh                Linux / macOS / Git Bash launcher for this lesson
      src/                    Code, requirements.txt, data, and runtime assets
    class.../
      readme.md
      start.bat, start.sh
      *.pptx                  Existing slides, where available
      src/
        *.py
        requirements.txt
        data/, pic/, runs/    Resources used by the individual lesson
  package1/                   Existing BUSI segmentation experiment, unchanged layout
  package2/                   Existing BreastMNIST classification experiment, unchanged layout
```

Lesson images, datasets, model weights, and existing run records are stored
alongside the code in each lesson's `src/` directory. Cross-lesson references in
Classes 2, 3, 4, 5, and 6 use the new paths. Existing lesson numbers are preserved;
missing lessons are not created.

## Set Up Python

Install Python 3.12, then run setup **once from the repository root**.
In Windows PowerShell:

```powershell
.\start.bat install
cd .\study\class8_cnn_playground
.\start.bat
.\start.bat app
```

On Linux / macOS / Git Bash:

```sh
sh start.sh install
cd study/class8_cnn_playground
sh start.sh
sh start.sh app
```

The root `install` command creates or reuses the repository's `.venv` and installs
dependencies for **all 12 lessons and package1/package2** in one pip operation.
`src/requirements.txt` includes the individual requirement files, so their
constraints are resolved together. The first installation needs network access
unless the packages are available in your pip cache or configured package source.
Machine learning packages can require substantial download and disk space.

This is a **project-wide virtual environment**, not a system-wide Python
installation. Every class launcher automatically uses the same `.venv`; you do
not need to activate it or install dependencies separately for each class.
Rerunning root `install` reuses the environment and checks the requirements again.
The command does not start training or download datasets. Installation failures
return a nonzero exit code; fix the reported issue and rerun setup.

There is no class-local `install` action. Root `start.bat` / `start.sh` without
arguments only display setup help; class launchers without arguments still
display their own lesson help. If setup has not been run, launchers fall back to
the available Python interpreter, which must already have any required packages.

GUI programs use Tkinter and require a desktop environment. Some Linux
distributions require a separate system installation of `python3-tk`.
Some also require `python3-venv` to create virtual environments.
Classes 3, 4, and 6 default to the Keras PyTorch backend and do not require
TensorFlow. If you set `KERAS_BACKEND` yourself, install the corresponding backend.

Normally, leave `PYTHON` unset so launchers select the shared environment
automatically. If needed, set `PYTHON` to a full interpreter path without
additional arguments. This overrides the interpreter for launching programs;
root `install` still installs into the repository's `.venv`, using the selected
interpreter to create it if necessary. Examples from the repository root:

```powershell
$env:PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\study\class13_house_price_regression\start.bat training --epochs 20
```

```sh
PYTHON="$PWD/.venv/bin/python" sh study/class13_house_price_regression/start.sh training --epochs 20
```

## Launch Each Lesson Independently

Each `class` directory has its own `start.bat` and `start.sh`.
**No lesson number is required.** Enter the lesson directory, then run:

```powershell
.\start.bat
.\start.bat --help
.\start.bat <action-or-entry-point-filename> [program-arguments...]
```

On Linux/macOS, replace `.\start.bat` with `sh start.sh`; the arguments are
identical. **Without arguments, the launcher only displays this lesson's help**,
including available actions and entry-point filenames. It does not install
dependencies, open a GUI, or train a model. `help`, `--help`, and `-h` also display
the lesson's help.

`train` is an alias for `training`. Entry-point filenames are also accepted:
for example, run `.\start.bat train_model.py --epochs 20` inside Class 13.

Launchers automatically use the lesson's `src/` as the working directory.
You can also invoke a launcher by path from the repository root or another
working directory:

```powershell
.\study\class13_house_price_regression\start.bat training --epochs 20
```

**Relative paths passed to a program are resolved from that lesson's `src/`
directory.** Use absolute paths for resources elsewhere. Program error exit
codes are returned unchanged.

| Lesson | Actions and entry points | Windows example (inside the lesson directory) |
|---|---|---|
| Class 1: Dataset preparation | `app`: `dataset_app.py` | `.\start.bat app` |
| Class 2: Dataset splitting | `split`: `split_dataset.py` | `.\start.bat split` |
| Class 3: Keras digit classification | `training`: `train_model.py` | `.\start.bat training` |
| Class 4: Model prediction | `app`: `app.py`; `predict`: `inf_model.py` | `.\start.bat app` / `.\start.bat predict` |
| Class 5: PyTorch training loop | `training`: `train_model.py` | `.\start.bat training` |
| Class 6: Evaluation demonstrations | `accuracy`, `metrics`, `confusion`: corresponding `test_*.py` files | `.\start.bat test` (runs all three) |
| Class 7: Feature extraction | `training`: `train_model.py`; `cnn`: `train_cnn.py` | `.\start.bat training` / `.\start.bat cnn` |
| Class 8: Interactive CNN | `app` / `training`: `app.py` | `.\start.bat app` |
| Class 9: Overfitting and generalization | `generate`: `generate_data.py`; `training`: `train_model.py` | `.\start.bat training --epochs 20` |
| Class 13: House-price regression | `generate`, `training`, `predict`: corresponding `.py` files | `.\start.bat predict --area 120 --rooms 3` |
| Class 14: LSTM forecasting | `generate`: `generate_data.py`; `training`: `train_model.py` | `.\start.bat training --epochs 20` |
| Class 15: Animal labels and centers | `app`: `pic/app.py`; `training`: `train.py`; `mtrain`: `mtrain.py`; `test`: `test.py`; `check`: unit tests | `.\start.bat mtrain`; `.\start.bat test m` (or `1` / `2`) |

Class 2 recreates its `src/train`, `src/val`, and `src/test` directories. Back up
custom data first. Training in Classes 3, 5, and 7 overwrites existing models and
training plots with the same names. Classes 4 and 6 read Class 3's model by
default, and digit classification data comes from Class 2. To prepare everything
again, run Classes 1, 2, and 3 in order, followed by Class 4 or 6.
Class 6 also uses the existing `weights_underfitting.h5`; normal Class 3 training
does not regenerate that file.

Class 8 starts training inside the GUI. Its `training.py` is an imported module,
not a standalone command-line training entry point. The first training run may
download CIFAR-10 and requires network access and storage.

Classes 9, 13, and 14 do not need data regeneration when data is already present.
Their generators refuse to overwrite existing data unless you explicitly pass
`--overwrite`. Do not regenerate data to improve test results. New training
records are saved under each lesson's `src/runs/`.

For programs that support command-line options, inspect their help from the
repository root:

```powershell
.\study\class9_overfitting_generalization\start.bat training --help
.\study\class13_house_price_regression\start.bat training --help
.\study\class14_lstm_forecasting\start.bat generate --help
```

## Automated Tests and Evaluation Programs

Run root setup once, then invoke lesson tests separately from the repository
root to avoid conflicts between identically named Python modules:

```powershell
.\study\class8_cnn_playground\start.bat test
.\study\class9_overfitting_generalization\start.bat test
.\study\class13_house_price_regression\start.bat test
.\study\class14_lstm_forecasting\start.bat test
.\study\class15_multi_tasks_learning\start.bat check
```

Each command runs `python -m unittest discover -s . -p "test_*.py" -v` in the
lesson's source directory. Class 14's published-result reproduction test uses
existing run records and is skipped if no published run exists.

Class 15's `test` action evaluates saved models on `src/pic/data3`, rather than
running code checks. Run `training 1` and `training 2` first to create `src/1.model`
and `src/2.model`, then use `test 1`, `test 2`, or `test` (both). Reports include
classification metrics, confusion matrices, and center errors. Full-test
and known-class-only scores are separate because those models learn three classes.
Alternatively, `mtrain` trains all six classes on data1 + data2 and saves
`src/m.model`; `test m` evaluates that combined model on data3.
Use `test compare` to evaluate all three models together and save comparison
charts under `src/runs/evaluation`.
All Class 15 training modes use a seeded, stratified 80/20 train/validation
split and report both sets' metrics. data3 is reserved for testing.
Each mode uses 80 training and 20 validation images: `mtrain` selects 40 train
and 10 validation images from each source rather than using all 200 images.
The current data3 contains ten animal types outside the six training classes:
classification accuracy is therefore zero for all three models; center errors
remain measurable and known-class-only scores are N/A.

Class 6 provides actual model evaluation demonstrations rather than `unittest`
tests. Inside its lesson directory, run all demonstrations or select one:

```powershell
.\start.bat test
.\start.bat accuracy
.\start.bat metrics
.\start.bat confusion
.\start.bat test_tfpn.py
```

Confusion matrix plots are saved to `study\class6_metrics\src\samples\`.
Other lessons currently have no automated test action; use their training,
prediction, or GUI programs for experiments.

The launcher tests require no machine learning dependencies. Run them from the
repository root:

```powershell
python -m unittest discover -s .\src -p "test_*.py" -v
```

You can also bypass the launcher and run programs directly from a lesson's
source directory. In this case, first activate the shared environment from the
repository root:

```powershell
.\.venv\Scripts\Activate.ps1
cd .\study\class13_house_price_regression\src
python train_model.py --epochs 20
python predict.py --area 120 --rooms 3
python -m unittest discover -s . -p "test_*.py" -v
```

On Linux/macOS, activate with `. .venv/bin/activate` before entering a lesson
source directory. Activation is only needed for direct `python` commands, not
for the class launchers.

## package1 / package2 Experiments

Both existing packages retain their directory layouts and do not use the lesson
startup scripts. Their dependencies are already included in root setup.
Activate the shared environment, enter the relevant package directory, and run
its programs directly. For example, starting from the repository root after setup:

```powershell
.\.venv\Scripts\Activate.ps1
cd .\package2
python train_breast_cancer.py --prepare-data
python train_breast_cancer.py --epochs 1 --output-dir outputs\lesson_smoke
python demo_app.py
python -m unittest discover -s . -p "test_*.py" -v
```

package2 downloads the official dataset only when `--download` is explicitly
provided, for example:
`python train_breast_cancer.py --prepare-data --download`.
The `unittest` command above runs offline unit tests. To evaluate a trained model
on the independent test split, use
`python train_breast_cancer.py --test --weights <weights-path> --output-dir <output-directory>`.

For package1, use the same activated environment, enter its directory, and run
`python train_breast_cancer.py --help` to inspect training options. No additional
package-specific installation is needed.
See each package's `readme.txt` for model training, prediction, and data requirements.

The breast ultrasound experiments and synthetic house-price and price-series
lessons are educational only. They are not intended for medical diagnosis,
real property valuation, or investment decisions.
