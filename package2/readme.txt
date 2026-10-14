BreastMNIST64 Binary Classification Learning Package
A beginner's guide for high-school students
===================================================

1. What are we learning?

This model answers one image-level question: normal_or_benign or malignant?
It does not draw a lesion mask. Unlike package1's U-Net segmentation lesson,
this lesson uses a small classifier with two convolution blocks.
No clinical use: a model score is NOT a diagnosis or calibrated clinical risk.
All reported metrics count images, never patients.

The official labels are 0=malignant and 1=normal or benign.
We reverse labels exactly once: package_label = 1 - official_label.
Our classes are 0=normal_or_benign and 1=malignant (the positive class).
Normal and benign images are merged; this is not a three-class task.

2. Files and data

  train_breast_cancer.py        One script for prepare/train/test/predict
  demo_app.py                  Desktop app: automatically classify example image
  requirements.txt             numpy, Pillow, torch>=2.0, matplotlib
  test_train_breast_cancer.py   Offline standard-library unittest tests
  data\DATASET_INFO.txt         Source, license, checksum, citations
  data\breastmnist_64.npz       Official archive (download only on request)
  example_ultrasound.png        Teaching example from the training split
  ML_FOR_HIGH_SCHOOL.md         Visual lesson and slide storyboard
  teaching_assets\              Teaching diagrams/reference illustrations
  make_teaching_images.py       Teaching image generator

ML_FOR_HIGH_SCHOOL.md contains 32 English slide outlines; teaching_assets\
contains six SVG diagrams. make_teaching_images.py uses Windows Arial.
Generated illustrations use training-reference images, not independent tests.

The example image is NOT an independent test image and must not be used to
claim held-out performance. Only --test evaluates the official held-out split.

BreastMNIST64 provides grayscale uint8 images (64x64) and image labels:
546 training images, 78 validation images, 156 test images.
There are no masks and no new random split. All training/validation images
are used. The official test split is kept out of training and model selection.
The archive checksum is validated every time data is loaded.
Data license: CC BY 4.0; attribution/citations are in data\DATASET_INFO.txt.

3. Quickstart in Windows PowerShell

These commands assume you first enter the package folder:
  cd .\package2

If dependencies are already installed, use your existing Python interpreter.
Otherwise, create an isolated environment:
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
Replace "python" below with ".\.venv\Scripts\python.exe" to use that environment.

Download/validate the official archive and report counts, without training:
  python .\train_breast_cancer.py --prepare-data --download
If the archive is already included, no network is needed:
  python .\train_breast_cancer.py --prepare-data

One short training experiment:
  python -u .\train_breast_cancer.py --epochs 1 --output-dir .\outputs\lesson_smoke
A 10-epoch experiment (starts fresh, never resumes the first experiment):
  python -u .\train_breast_cancer.py --epochs 10 --output-dir .\outputs\my_first_run

Evaluate the saved checkpoint ONCE, after choosing settings using validation:
  python .\train_breast_cancer.py --test --weights .\outputs\my_first_run\breastmnist_cnn.pt --output-dir .\outputs\test_my_first_run

Demonstrate one image, saving results without opening a plot window:
  python .\train_breast_cancer.py --predict .\example_ultrasound.png --weights .\outputs\my_first_run\breastmnist_cnn.pt --output-dir .\outputs\my_demo --no-show
Omit --no-show to also display the prediction plot.
Prediction needs a checkpoint and image, but does not access/download data.

Show options:
  python .\train_breast_cancer.py --help
Open the desktop demonstration (automatically predicts example_ultrasound.png):
  python .\demo_app.py
Use another image or checkpoint:
  python .\demo_app.py --image .\example_ultrasound.png --weights .\outputs\my_first_run\breastmnist_cnn.pt
The window also has Choose image, Load weights, and Predict buttons.
It uses the same grayscale resize, inference, label mapping, and saved threshold
as the command-line predictor, with CPU inference and no training/downloads.
It shows dataset class (normal/benign or malignant), NOT lesion presence.
The included default example is malignant by its reference label, but the
included short-run model predicts normal/benign. This is a real model mistake.
The UI displays educational warnings; a low score does not rule out cancer.
Only load trusted checkpoints. File/model failures show an error dialog and log
the exception instead of leaving a stale successful prediction on screen.
The app needs standard Python Tkinter and Pillow ImageTk; no web framework or
new third-party dependency is required. Standard Windows Python includes Tkinter.

Run offline unit tests from the repository root:
  python -m unittest discover -s .\package2 -p 'test_*.py'

Included example artifacts:
  outputs\first_try\       10 epochs, batch 32, lr 0.001, seed 42, CPU
  outputs\test_first_try\  Separate evaluation of the selected epoch-10 model
  outputs\demo\            Single training-image prediction demonstration
These folders already contain results; choose new folders for your own runs.
The example test accuracy is 73.1%, but malignant Recall is only 4.8%
(TP=2, FP=2, FN=40, TN=112). This intentionally transparent short-run example
illustrates why accuracy alone is misleading, not a successful cancer detector.
Do not tune the model repeatedly against these test results.

4. Defaults and experiment choices

Default mode: train. --prepare-data, --test, and --predict IMAGE are mutually
exclusive alternatives. Training does NOT evaluate test images automatically.
--weights is required for test/predict and forbidden during training.
--no-show is prediction-only. --download permits the official download only
when data is missing (not allowed for prediction).

  --epochs       10       Positive number of passes through training images
  --batch-size   32       Positive images per batch
  --lr           0.001    Finite positive Adam learning rate
  --seed         42       Integer between 0 and 4294967295
  --workers      0        Nonnegative; keep 0 initially on Windows
  --threads      2        Positive number of PyTorch CPU computation threads
  --threshold    0.5      Training-only, finite in [0,1]; saved in checkpoint
  --device       auto     auto, cpu, or cuda (cuda must be available)
  --data-dir              data under this script's package folder
  --output-dir            outputs\BreastMNIST64 under this script's folder

The input size is fixed at 64; there is no --image-size setting.
Custom relative paths are relative to the terminal's working directory.
Default paths remain relative to the script even when launched from elsewhere.
--data-dir contains breastmnist_64.npz, not extracted image folders.
Use a new output directory for every experiment and for its separate test.
If any known result artifact already exists, the script refuses to run unless
you explicitly choose --overwrite. This overrides only files written by the
current mode; unrelated files/old artifacts from other modes are not deleted.

Change one training setting at a time. Use validation loss for choices.
Repeatedly tuning settings after looking at test results spoils the final exam.
The random seed helps reproducibility, but does not promise identical results
across different devices, library versions, or hardware.

5. Model and learning

Inputs: [batch, 1, 64, 64], normalized only by uint8 / 255.
External images are converted to grayscale and resized directly to 64x64
with bilinear interpolation. Their aspect ratio can change; this simple
demonstration is not a validated external-image clinical pipeline.

SimpleCNN:
  Conv2d(1,8,3,padding=1) -> ReLU -> MaxPool2d(2)
  Conv2d(8,16,3,padding=1) -> ReLU -> MaxPool2d(2)
  AdaptiveAvgPool2d((4,4)) -> Flatten
  Linear(256,32) -> ReLU -> Linear(32,1)

This model has 9,505 trainable parameters. At batch size 32, the 546 training
images form 18 batches per epoch; the last batch contains 2 images.

The model returns one logit per image, shape [batch]. BCEWithLogitsLoss
compares logits with package labels; Adam updates weights during training.
Sigmoid turns each logit into a malignant model score between 0 and 1.
Score >= threshold predicts malignant; equality counts as positive.
No augmentation, batch normalization, dropout, torchvision, sklearn,
medmnist dependency, masks, heatmaps, or lesion-localization claims.

6. Saved results

Training:
  breastmnist_training_metrics.csv
    epoch,train_loss,train_accuracy,val_loss,val_accuracy,val_precision,
    val_recall,val_specificity,val_f1
    One row per completed epoch; undefined metrics are blank.
  breastmnist_cnn.pt
    Saved immediately whenever validation LOSS improves (not accuracy).
    Metadata: architecture,dataset,image_size,label_mapping,class_names,
    threshold,best_epoch,best_validation_loss,settings,dataset_md5,
    split_counts,state_dict.
    settings: epochs,batch_size,lr,seed,workers,threads,device,image_size,
    optimizer,loss,normalization.
    Loading uses torch.load(weights_only=True), verifies compatibility and
    the label mapping, and strictly checks state_dict. No silent resume.
  learning_curves.png
    Actual measured training/validation loss and accuracy by epoch.
  validation_confusion_matrix.png
    Generated at the end of every successful training run. Uses validation
    predictions from the lowest-validation-loss checkpoint, NOT test data
    or necessarily the final epoch. Rows=true, columns=predicted:
    [[TN,FP],[FN,TP]]. The title includes the best epoch and saved threshold.
  training_summary.json
    dataset,dataset_md5,seed,epochs,best_epoch,best_validation_loss,threshold,
    best_validation_metrics,label_names,label_mapping,split_counts,settings,
    notice. best_validation_metrics includes loss, accuracy, precision, recall,
    specificity, f1, TP/FP/FN/TN, and total used for the matrix. No test metrics.

Standalone test:
  test_metrics.json
    loss,accuracy,precision,recall,specificity,f1,tp,fp,fn,tn,total,dataset,
    dataset_md5,threshold,label_names,label_mapping,weights_provenance,
    unit ("image"),notice.
  test_predictions.csv
    index,true_label,predicted_label,malignant_score (official test order).
  confusion_matrix.png
    Rows=true label, columns=predicted label; [[TN,FP],[FN,TP]].
  test_examples.png
    Fixed first 8 test images with true class, prediction, and malignant score.

Single-image prediction:
  prediction.png
    Resized input with predicted class/score and explicit not-diagnosis notice.
  prediction.json
    input_image,predicted_label,predicted_class,malignant_score,threshold,
    label_names,label_mapping,weights_provenance,notice.

weights_provenance (test/predict) records checkpoint absolute path, MD5,
architecture,best_epoch,best_validation_loss,training_settings.
Test and prediction always use the saved threshold, not a new CLI threshold.
Checkpoint MD5 identifies the weights file; it is not an authenticity guarantee.

7. How to read binary image metrics

TP: malignant image predicted malignant; FN: malignant predicted nonmalignant.
FP: normal/benign predicted malignant; TN: normal/benign predicted nonmalignant.
  accuracy    = (TP+TN)/total
  precision   = TP/(TP+FP)
  recall      = TP/(TP+FN)
  specificity = TN/(TN+FP)
  F1          = 2*TP/(2*TP+FP+FN)
Loss is mean image binary cross-entropy, not a percentage.
Accuracy alone can mislead when class counts differ.
A zero denominator makes a metric undefined: JSON null, CSV blank, and printed
N/A with an explanation. Do not replace it with a fabricated zero or perfect one.

8. Errors, interruptions, and responsible use

Missing dataset: explicitly run --prepare-data --download.
Bad checksum: the script raises an error and does not silently replace the file.
Move/remove only the invalid archive yourself, then explicitly download again.
Downloads use a unique partial file beside the destination, verify its checksum,
and replace the destination atomically. Failed partial downloads are removed,
and network/checksum exceptions propagate rather than hiding failure.
Incompatible checkpoint: use this classifier's checkpoint, not package1's U-Net.
Unavailable CUDA: choose --device cpu. Invalid numeric options fail immediately.
Existing results: pick another --output-dir, or intentionally use --overwrite.
Memory problems: reduce --batch-size and keep --workers 0.

Per-epoch CSV rows are flushed and best weights are saved during training.
Interruptions may leave an incomplete experiment (plots, including the
validation confusion matrix, and summary are created
only after all epochs finish). Training always starts from scratch; this is
not a resume system. Avoid concurrent runs sharing an output directory.
Only load trusted checkpoint files, even with the restricted loader.

Small benchmark image sets cannot demonstrate screening performance, clinical
safety, patient-level generalization, or calibrated cancer probabilities.
Do not submit personal medical images or use predictions for medical decisions.
