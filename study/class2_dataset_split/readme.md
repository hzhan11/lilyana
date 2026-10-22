# Class 2: Dataset Splitting

## Before You Start

A model can memorize examples without learning a rule that works on new ones.
Testing it on the same examples used for learning is like giving a student an
exam containing only questions they have already practiced.

We therefore separate a dataset into three groups. The **training set** is used
to update the model's learned numbers. The **validation set** helps us compare
settings and monitor learning without updating those numbers from its labels.
The **test set** is reserved for evaluating the chosen model after those
decisions are finished. **Generalization** means performing well on genuinely
new examples, not just the training set.

### What This Experiment Shows

The script copies Class 1's images into training, validation, and test folders
using a 60% / 20% / 20% split. A fixed **random seed** makes the split repeatable
when the input files are unchanged. It does not make the split automatically
representative or guarantee the same number of each digit in every group.

### What to Look For

After running the script, inspect the three folders and the printed counts.
Each original sample should appear in only one group. Do not move difficult
test examples into training to improve a score: that would make the evaluation
misleading. Even different files can be near-duplicates, so separation by filename
alone does not guarantee independence. With only a few test examples, one mistake
can change the reported accuracy substantially. This lesson uses a random split;
time-ordered forecasting data needs a different approach, as Class 14 explains.

(code understanding is recommended)

## Running

First run `.\start.bat install` (Windows) or `sh start.sh install` once from the
**repository root**. All lessons share its `.venv`; no per-class installation
or environment activation is needed when using the launchers.

Code and datasets are in `src/`. From this class directory (no arguments shows help only):
`.\start.bat split`.
This recreates `src/train`, `src/val` and `src/test` from Class 1's `src/png`.
On Linux/macOS use `sh start.sh` instead of `start.bat`.
See the [execution guide](../../readme.md).

## Homework

**Try a different split (3-line edit).** Back up `src/train`, `src/val`, and
`src/test` first: the splitter replaces them. In `src/split_dataset.py`, change
the three ratio constants to:

```python
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
```

Run `.\start.bat split` and compare the printed counts with the original split.
Which groups became smaller when training received more examples?
Restore the original constants and backed-up folders before continuing to other
classes; do not use this exercise to choose a split based on test scores.

## Page 1: Training, Validation, and Test Sets

![Page 1: Training, Validation, and Test Sets](src/pic/dataset_split.png)

## Page 2: Test Sample

![Page 2: Test Sample](src/test/2_03.bmp)
