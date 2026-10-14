BUSI Ultrasound Segmentation Learning Package
A beginner's guide for high school students
============================================================

English visual lesson and PPT storyboard:
  ML_FOR_HIGH_SCHOOL.md
  32 slide outlines, speaker notes, calculations, and optional mini-labs.
  Uses the actual segmentation project and distinguishes pixel-level
  metrics from patient-level cancer-screening claims.
  teaching_assets\ contains SVG diagrams and reference-image illustrations.
  make_teaching_images.py regenerates the two reference illustrations on
  Windows using the installed project dependencies and the Arial font.
  The Markdown is a lesson handout/storyboard, not an exported PPTX file.

1. What does the computer learn?

Think of this task as coloring in the lesion in an ultrasound picture.
The images folder contains ultrasound pictures. The labels folder contains
matching answer pictures, called masks, with the same filenames.
The program treats nonzero mask pixels as lesions and zero pixels as background.
U-Net learns to predict "lesion or background" for every pixel.

Important: This is NOT a benign/malignant classifier or a medical diagnostic tool.
"Benign" means noncancerous; "malignant" means cancerous. Here these names help
balance the dataset splits. They are not the model's two output classes.
The downloaded subset does not contain the normal category.

2. Where are the files?

package1\
  train_breast_cancer.py       Training program with Chinese teaching comments
  requirements.txt            Required Python libraries
  readme.txt                  This guide
  data\BUSI\
    BUSI\images\              629 pictures: 420 benign and 209 malignant
    BUSI\labels\              629 matching lesion masks
    BUSI.zip                  Downloaded archive; already extracted
    DATASET_INFO.txt          Source, license, and citation information

By default, the model and results table go into this package's outputs\BUSI\.
This copy does not change the original project's script or dataset.
There is no need to download the dataset again.

3. First run: Windows PowerShell

Enter. Alternatively, from a terminal in the original project folder, run:
  cd .\package1

Create a separate Python environment to avoid changing other projects:
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Start with a short experiment to check that everything works:
  .\.venv\Scripts\python.exe -u .\train_breast_cancer.py --epochs 1 --image-size 64 --batch-size 4 --output-dir .\outputs\first_try

A small experiment suitable for learning on a CPU:
  .\.venv\Scripts\python.exe -u .\train_breast_cancer.py --epochs 10 --image-size 128 --batch-size 4 --output-dir .\outputs\cpu_10

A 100-epoch experiment (may take a long time; try a short run first):
  .\.venv\Scripts\python.exe -u .\train_breast_cancer.py --epochs 100 --image-size 128 --batch-size 4 --output-dir .\outputs\cpu_100

If your existing Python environment already has the required libraries, you
can replace .\.venv\Scripts\python.exe with python. The -u option makes printed
output appear promptly.
You can also run the script in PyCharm. Without command-line options, it uses
the defaults below. Do not run a full 100-epoch experiment in the debugger.

4. How do I change the parameters?

Add --parameter-name value at the end of the command; no code changes are needed.
Change one parameter at a time and choose a new output directory so that you
can compare experiments fairly.

--epochs          Default: 20
  Number of training rounds. One epoch passes through the training set once.
  Try 1 or 10 before trying 100. More epochs do not always improve the model:
  it may simply memorize the training pictures. This is called overfitting.

--batch-size      Default: 2
  Number of pictures processed together. Try 2, 4, or 8.
  Larger batches generally need more RAM or GPU memory.
  Reduce this first if memory runs out. It can also affect learning behavior.

--image-size      Default: 256
  Side length of the square model input. Pictures keep their aspect ratio,
  then receive black padding. The original files are not changed.
  Must be at least 32 and a multiple of 16: for example, 64, 128, 256, or 512.
  Start with 128 on a CPU. Larger inputs retain more detail but are slower
  and need more memory. Low-resolution results do not establish performance
  at the original image resolution.

--lr              Default: 0.0001 (also written as 1e-4)
  Learning rate: roughly, how big a step the model takes when correcting itself.
  If loss fluctuates strongly, compare 0.00005. If learning is slow, compare
  0.0002. No value is guaranteed to be best. Keep other settings unchanged.

--val-fraction    Default: 0.15
--test-fraction   Default: 0.15
  Fractions reserved for validation and testing. Both must be greater than
  zero, and their sum must be less than 1. The remainder is used for training.
  Defaults give 441 training, 94 validation, and 94 test pictures.
  Counts are rounded separately for each category, so proportions vary slightly.
  Think of training as practice questions, validation as a mock exam, and
  testing as the final exam. Repeatedly adjusting settings after checking test
  results turns the "final exam" into another practice exercise.

--seed            Default: 42
  Controls random operations such as shuffling and model initialization.
  Keep it fixed when comparing parameters. It does not guarantee identical
  results across different devices.

--workers         Default: 0
  Number of subprocesses used to load data. Beginners on Windows should keep
  this at 0. Later, compare 2. More workers are not always faster and use
  additional resources.

--data-dir        Default: data\BUSI under the script's folder
  Must point to the folder containing BUSI\images and BUSI\labels,
  not directly to the images folder.

--output-dir      Default: outputs\BUSI under the script's folder
  Where results are saved. Custom relative paths are interpreted relative
  to the terminal's current working directory.
  Example: --output-dir .\outputs\experiment2

List all available options:
  .\.venv\Scripts\python.exe .\train_breast_cancer.py --help

5. What do the numbers mean?

Results are printed once at the end of each epoch, not after every picture:
  train loss: Training error measure; generally, lower is better.
  val loss: Validation error measure; generally, lower is better.
  train/val pixel acc: Pixel accuracy; 0.95 means 95% of pixels were correct.
  val Dice: Overlap between predicted and actual lesions; closer to 1 is better.

Loss combines pixelwise binary cross-entropy (BCE) loss and Dice loss.
It is not a percentage.
Pixel accuracy includes background and black padding. It can look high even
when lesion segmentation is poor. It does NOT mean "95% of patients were
diagnosed correctly." Always consider Dice alongside pixel accuracy.
The best model is selected by validation Dice, not by training accuracy.
At the end, Test pixel accuracy and Test Dice describe that selected model.

6. Results, interruptions, and common problems

busi_training_metrics.csv: Per-epoch results; open it in Excel to plot curves.
busi_unet.pt: Model weights and settings such as input size.
It is not a program you can run by double-clicking.

The CSV is updated every epoch, but this version saves model weights only
after all epochs finish. Closing the terminal, a power failure, or pressing
Ctrl+C can lose the current weights; sleep can pause or disrupt the run.
The CSV cannot be used to resume training. Every run starts from scratch.
Reusing an output directory overwrites the CSV, and saving the finished model
overwrites an existing model with the same filename.
Use a different output directory for each experiment. Avoid running several
large training jobs at the same time.

The program uses CUDA if it is available; otherwise, it uses the CPU.
CPU training can be slow without being stuck. Look for new epoch output
and updates to the CSV.
If pictures cannot be found, check the folder structure.
If torch or another module cannot be found, install requirements using the
same Python environment that runs the script.
If memory runs out, reduce batch-size or image-size.

7. Data sources and scientific limitations

This package uses a curated BUSI subset, not the complete original BUSI dataset.
Source: https://zenodo.org/records/21128640
License: CC BY 4.0. Preserve attribution and describe changes when sharing.
Original dataset paper: https://doi.org/10.1016/j.dib.2019.104863
See data\BUSI\DATASET_INFO.txt for additional information.

The program splits data by image, not by patient. Pictures from the same
patient or very similar pictures could leak information between splits.
These test scores are not evidence of clinical reliability.
This package is for education and research only.

8. Suggested order for reading the code

main: Reads and checks your parameter values.
find_samples: Matches each picture with its answer mask.
stratified_split: Divides pictures into training, validation, and test sets.
BUSIDataset: Reads, resizes, and flips pictures, then converts them into tensors.
UNet: Defines the model that extracts features and predicts lesion locations.
run_epoch: Processes one round of data; during training, computes gradients
and updates model parameters.
train: Organizes the epochs, selects the best model, tests it, and saves it.

9. Predict and display a single image (no training)

example_ultrasound.png in the package1 root is a copy of benign_148.png
from the supplied dataset. It is an illustration, not an independent clinical
test. Whether an image belongs to a training or test split depends on the seed
and split fractions used for that training run.

Install the updated requirements first to get matplotlib:
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt

From package1, load the weights produced by your first_try run:
  .\.venv\Scripts\python.exe .\train_breast_cancer.py --predict .\example_ultrasound.png --weights .\outputs\first_try\busi_unet.pt

For a model trained with the default output directory:
  .\.venv\Scripts\python.exe .\train_breast_cancer.py --predict .\example_ultrasound.png

--predict switches from training to single-image prediction.
--weights selects the saved model; it defaults to busi_unet.pt in output-dir.
The script uses the input size and threshold stored in those weights, not the
--image-size training option. It does not train a new model.

A window displays the original image, a predicted lesion mask, and a red
overlay. Close the window when finished. The plot and a binary mask are also
saved in output-dir as <image-name>_prediction.png and <image-name>_mask.png.
For a machine without a graphical display, add --no-show to save files only.
You may supply another image path with --predict.

The red region is a predicted lesion location, not a benign/malignant diagnosis.
A single image without a reference mask does not provide an accuracy score.
This mode does not compare with the dataset's ground-truth mask.
It uses the best-validation model saved after training, not necessarily the
model from the final epoch. A short first_try model may predict poorly.
