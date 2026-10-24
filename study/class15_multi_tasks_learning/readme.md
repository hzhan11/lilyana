# Class 15: Animal Classes and Center Coordinates

## Before You Start

An image can have more than one label. Its animal name is a **classification**
target, while its center `[x,y]` is a numerical **localization** target. A
multi-task neural network learns to predict both from the same image. A shared
CNN extracts features, then two separate output heads learn the animal class
and center coordinates. `train.py` trains this network; the viewer app still
only displays ground-truth labels, not model predictions.

Image coordinates start at the top-left corner. The x coordinate increases to
the right, and y increases downward. The supplied images are 200 by 200 pixels,
so valid pixel coordinates range from 0 to 199.

## Running

Install shared dependencies once from the **repository root** with
`.\start.bat install` on Windows or `sh start.sh install` on Linux/macOS.
Then, from this class directory:

```powershell
.\start.bat app
```

On Linux/macOS use `sh start.sh app`. No arguments display the lesson help.
A desktop environment and Tkinter are required.

The app itself is at `src\pic\app.py`. To run it directly from the repository
root with the shared Windows environment:

```powershell
.\.venv\Scripts\python.exe .\study\class15_multi_tasks_learning\src\pic\app.py
```

## Browsing Datasets

The viewer discovers `data*` folders next to `app.py`, including `data1`, `data2`,
and `data3`. Each dataset uses its own `label.txt`, with images in the same folder:

```text
image_01.png,dog,[82,100]
image_02.png,pig,[130,100]
```

`data1` contains 100 cat/dog/pig images; `data2` contains 100 rabbit/bear/fox
images. Their original ten images are preserved; each has 90 additional
randomly recolored, resized, rotated, and positioned animal drawings.
`data3` contains ten randomly colored and positioned drawings of **unseen
animals**: elephant, lion, turtle, bird, fish, cow, horse, sheep, zebra, and
giraffe (one each). None belongs to the six training classes. All three datasets
use 200-by-200 PNGs and the same center-label format. The viewer accepts both
the six training classes and these ten new labels.

Select a folder to load its labels, then select an image or use **Previous** and
**Next**. A red circle and crosshair show its labeled center. The details line
shows the filename, animal class, original center coordinates, and image size.
Resizing the window scales the marker together with the image.

Use **Show center** to hide or display the marker, **Refresh folders** after
adding a new dataset, or **Open label file...** to browse another label file.
New dataset folders need a valid `label.txt` and matching images before they
can be loaded. Missing images, malformed labels, and
out-of-bounds coordinates produce an explicit error and clear the previous view.
The viewer never changes the images or labels.

To open a specific folder at startup:

```powershell
.\start.bat app --data-dir pic\data1
```

Program arguments are relative to this class's `src/` directory. Use
`pic/data1` for the equivalent Linux/macOS command.

## Training

From this class directory, select one dataset:

```powershell
.\start.bat training 1
.\start.bat training 2
```

On Linux/macOS use `sh start.sh training 1` or `sh start.sh training 2`.
`train` is also accepted as an alias for `training`. Omit all arguments to
display launcher help; `.\start.bat training --help` shows training options.
Direct execution from `src` is `python train.py 1` or `python train.py 2`.

To train **one six-class model on data1 + data2 together**:

```powershell
.\start.bat mtrain
.\start.bat mtrain --epochs 20
```

On Linux/macOS use `sh start.sh mtrain`. Direct execution from `src` is
`python mtrain.py`. This selects 50 images from each dataset: 40 for training and
10 for validation, for a total of **80 train + 20 validation images**. It preserves
the separate labels for identically named files in the two folders and saves
**`src/m.model`**. The other 100 available images are unused for that run.
It trains a fresh model rather than merging the weights of `1.model` and
`2.model`. **data3 is never used for training.**
`mtrain` accepts the same optional training settings as `training 1` / `training 2`,
including `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--threads`,
and `--output-dir`.

### Train/Validation Split

All three training modes use a fixed **80% train / 20% validation** split:

| Training command | Training images | Validation images |
|---|---:|---:|
| `training 1` | 80 | 20 |
| `training 2` | 80 | 20 |
| `mtrain` | 80 (40 from each source) | 20 (10 from each source) |

The split is shuffled and stratified by animal class to keep class proportions
similar in both subsets. `--seed` controls the split and training randomness;
the same seed and unchanged label order reproduce the same memberships.
For `mtrain`, each source first uses the same seeded 80/20 partition as its
single-dataset baseline. It then selects 40 stratified training images from
that source's 80-image training pool and 10 stratified validation images from
its 20-image validation pool. This keeps image counts equal across all three
models and prevents an image from changing its train/validation role between
the baseline and combined model. A custom source needs at least 50 images for
combined training, with enough examples per class for stratification.
Only training batches receive gradient updates. Validation runs in evaluation
mode without gradients after every epoch. data3 remains separate for testing.
Custom datasets need enough samples for every class to appear in both subsets;
invalid stratified splits produce an explicit error.

Training defaults to 100 epochs on CPU. For a shorter run:

```powershell
.\start.bat training 1 --epochs 20 --batch-size 5 --learning-rate 0.001
```

`src/model.py` defines `AnimalCNN`. Each image is converted to **RGBA**, including
an alpha channel (opaque RGB PNGs receive alpha 255), and divided by 255.
Input is a float tensor shaped **(batch, 200, 200, 4)**. PyTorch's channel-first
conversion happens inside the network. The two outputs are:

- Class logits shaped `(batch, num_classes)`; `argmax` selects the animal.
- Center coordinates shaped `(batch, 2)` in original pixels, ordered `[x,y]`.
  A sigmoid constrains each coordinate to 0-199.

For data1, class indices 0-2 mean `cat`, `dog`, `pig`. For data2 they mean
`rabbit`, `bear`, `fox`. Each run trains a fresh three-class model; it does not
continue the other dataset's model. `mtrain.py` instead trains all six classes,
with indices 0-5 meaning `cat`, `dog`, `pig`, `rabbit`, `bear`, `fox`.
The CNN architecture and loss are shared between both training entry points;
only the number of classification outputs changes from three to six.

The loss is **cross-entropy + 10 * center MSE**, with both target and predicted
coordinates divided by 199 for the MSE calculation. This keeps pixel-sized
errors from overwhelming classification. Logs show separate train/validation
combined loss, accuracy, and center mean absolute error in pixels after each
epoch's updates (printed on epoch 1, every 10 epochs, and the final epoch).

Checkpoints are saved separately as **`src/1.model`**, **`src/2.model`**, and
**`src/m.model`**. Re-running a training mode replaces only its own checkpoint.
Each checkpoint contains model weights, ordered class names, input shape,
coordinate scale, training settings, split memberships and counts, unused
image indices, and per-epoch
train/validation metrics. History keys `loss`, `accuracy`, and `center_mae_pixels`
refer to training data; `val_loss`, `val_accuracy`, and `val_center_mae_pixels`
refer to validation data. Saved indices follow label-file order (data1 followed
by data2 for `mtrain`). The checkpoint still contains the **final epoch** model;
validation monitoring does not enable early stopping or select a different epoch.
Use `--output-dir` to choose another model directory. The `.model` extension is
a filename convention; these files are PyTorch checkpoint dictionaries.
Older `runs/data1/model.pt` / `runs/data2/model.pt` checkpoints are not loaded
automatically; retrain to create the new filenames.

To reload a checkpoint from a Python program in `src`:

```python
import torch
from model import AnimalCNN

checkpoint = torch.load("1.model", map_location="cpu", weights_only=True)
model = AnimalCNN(num_classes=len(checkpoint["classes"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
# images: normalized float RGBA tensor with shape (batch, 200, 200, 4)
with torch.no_grad():
    logits, centers = model(images)
animals = [checkpoint["classes"][index] for index in logits.argmax(dim=1).tolist()]
```

These datasets have 100 available images each (200 combined). Each training
mode uses 100 images in total, partitioned as shown above.
Training and validation scores are reported separately; neither is a data3
test score. Because augmented images share original drawing templates, an
image-level train/validation split is not an independent new-template benchmark.
Good scores do not demonstrate generalization to new drawings.

## Evaluating Models on data3

After training, run these commands from this class directory:

```powershell
.\start.bat test 1
.\start.bat test 2
.\start.bat test m
.\start.bat test
```

`test 1` loads `src/1.model`; `test 2` loads `src/2.model`; `test m` loads
`src/m.model`. For compatibility, `test` without a model identifier still
evaluates models 1 and 2 only; use `test m` explicitly for the combined model.
On Linux/macOS use the same arguments with `sh start.sh`.
The standalone entry point from `src` is `python test.py 1`, `python test.py 2`,
or `python test.py m`. `evaluate.py` remains available as the underlying
evaluation program. Missing models produce an error asking you to train first;
evaluation never trains a replacement automatically.

Evaluation reads `src/pic/data3/label.txt`, uses the same RGBA preprocessing as
training, and runs the saved model in evaluation mode without updating weights.
It reports:

- Accuracy; macro and support-weighted precision, recall, and F1.
- Per-class precision, recall, F1, and support (actual image count).
- A confusion matrix with actual classes in rows and predictions in columns.
- Center MAE and RMSE over x/y coordinates, plus mean Euclidean center distance
  in pixels. The JSON also contains separate x and y MAE values.

**All ten data3 animals are unseen by all three models.** Models 1 and 2 each
predict only three training classes; model m predicts only the six training
classes. None can output the new animal labels, so full-dataset accuracy,
precision, recall, and F1 are **zero by design**, not evidence of a loader
failure. Known-class-only results are **N/A** because there are no known-class
images. Center errors remain meaningful and are calculated for all ten images.

Full scores and confusion matrices include the union of the six training
classes and the unseen labels actually present in the test set (currently 16
labels), so no unseen ground-truth rows are dropped. If a custom test folder
contains known classes, a separate known-class-only report is also produced.
These are closed-set classifiers: they still predict one of their trained
classes, not an automatic "unknown" label. A high confidence on an unseen
animal does not make that classification correct.

Undefined precision, recall, and F1 are explicitly reported as zero.
Macro averages give each evaluated class equal weight; weighted averages use
its actual image count. If no known-class images exist, that group's results
are reported as N/A rather than fabricated zero metrics.

JSON reports, including every predicted animal, confidence, center, and
ground-truth label, are saved to `src/runs/evaluation/1.metrics.json` and
`src/runs/evaluation/2.metrics.json`; the combined model uses
`src/runs/evaluation/m.metrics.json`. Re-evaluation replaces the corresponding
report, not the model or images. Use `--model-dir`, `--data-dir`, and
`--output-dir` to choose different locations; `test --help` lists options.

### Comparing Models with Charts

To evaluate all three saved models on the same data3 images and save charts:

```powershell
.\start.bat test compare
```

On Linux/macOS use `sh start.sh test compare`. From `src`, direct execution is
`python test.py compare`. All three model files must exist; comparison never
trains missing models or substitutes previous saved scores. Existing `test 1`,
`test 2`, `test m`, and no-identifier behavior are unchanged.

Outputs are saved under **`src/runs/evaluation`**:

- **`comparison.png`**: full-dataset accuracy and macro precision/recall/F1,
  known-class-only scores, center MAE/RMSE/distance, and per-class F1.
- **`comparison_confusion.png`**: the three confusion matrices with identical
  class ordering and a shared color scale.
- **`comparison.metrics.json`**: the complete fresh reports used by the charts,
  alongside the updated individual `1.metrics.json`, `2.metrics.json`, and
  `m.metrics.json` reports.

The full-dataset panels use the same images for every model. Known-class-only
panels explicitly show their different sample counts; those scores are not
like-for-like full-test comparisons. Undefined known-class results appear as
N/A, not zero. Higher classification scores are better; lower center errors
are better. The charts are rendered headlessly and do not open a GUI.
Use `--output-dir` to change the chart/report directory. Re-running comparison
replaces these outputs without changing model weights or images.

This is a tiny synthetic out-of-distribution demonstration, not a benchmark
for ordinary known-animal classification. data3's drawings are newly generated
for different animal types, rather than recolored training animals. Center
localization can be compared across the models, but a useful known-class
classification benchmark would require a separate set of known-animal images.

## Automated Code Checks

From this class directory, run `.\start.bat check` or `sh start.sh check`.
Unlike `test`, `check` runs automated code tests and does not require trained
model files.
The tests cover label validation, RGBA conversion, both CNN outputs and their
gradients, training and checkpoint reloads, exact evaluation metrics, handling
unseen classes, coordinate scaling, navigation, and UI error handling. UI tests
require a Tk display and explicitly skip when one is unavailable. Training and
model evaluation do not require Tkinter or a display.

## Homework

**Try another marker color (1-line edit).** In `src/pic/app.py`, change:

```python
MARKER_COLOR = "#2563eb"
```

Run the app again. Does changing the marker color alter the coordinates shown
in the details line? Restore the original color afterward.
