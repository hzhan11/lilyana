from dataclasses import dataclass
import math
import tkinter as tk
from tkinter import ttk

import torch
from torch import nn

if __package__:
    from .training import CLASS_NAMES, Config, build_model
else:
    from training import CLASS_NAMES, Config, build_model


@dataclass(frozen=True)
class Stage:
    title: str
    shape: tuple[int, ...]
    details: tuple[str, ...]
    parameters: int = 0


def architecture_stages(config: Config) -> list[Stage]:
    # Meta tensors infer the real model's shapes without allocating its weights.
    with torch.device("meta"), torch.inference_mode():
        model = build_model(config).eval()
        value = torch.empty(1, 3, 32, 32)
        stages = [Stage("Input", (3, 32, 32), ("RGB image", "Channels x height x width"))]
        details, head, parameters, block = [], [], 0, 1
        for layer in model:
            value = layer(value)
            parameters += sum(parameter.numel() for parameter in layer.parameters())
            if isinstance(layer, nn.Conv2d):
                details.append(
                    f"Conv {layer.kernel_size[0]}x{layer.kernel_size[1]}: "
                    f"{layer.in_channels} -> {layer.out_channels}"
                )
            elif isinstance(layer, nn.MaxPool2d):
                details.extend(("BatchNorm + ReLU after each conv", f"MaxPool {layer.kernel_size}x{layer.kernel_size}"))
                stages.append(Stage(f"Block {block}", tuple(value.shape[1:]), tuple(details), parameters))
                details, parameters = [], 0
                block += 1
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                head.append(f"Global average pool: {value.shape[1]} x 1 x 1")
            elif isinstance(layer, nn.Flatten):
                head.append(f"Flatten: {value.shape[1]} features")
            elif isinstance(layer, nn.Dropout):
                head.append(f"Dropout: p = {layer.p:g}")
            elif isinstance(layer, nn.Linear):
                head.append(f"Linear: {layer.in_features} -> {layer.out_features}")
            elif not isinstance(layer, (nn.BatchNorm2d, nn.ReLU)):
                raise ValueError(f"Unsupported architecture layer: {type(layer).__name__}")
        stages.append(Stage("Classifier head", tuple(value.shape[1:]), tuple(head), parameters))
        stages.append(Stage(
            "Output", (len(CLASS_NAMES),),
            ("10 class scores (logits)", " / ".join(CLASS_NAMES[:2]),
             " / ".join(CLASS_NAMES[2:4]), " / ".join(CLASS_NAMES[4:6]),
             " / ".join(CLASS_NAMES[6:8]), " / ".join(CLASS_NAMES[8:])),
        ))
        return stages


class ArchitectureWindow(tk.Toplevel):
    CARD_WIDTH = 218
    CARD_HEIGHT = 232
    GAP_X = 42
    GAP_Y = 48
    MARGIN = 24

    def __init__(self, parent, config):
        self.stages = architecture_stages(config)
        self.config_snapshot = config
        super().__init__(parent)
        self.title("CNN Model Architecture")
        width = min(1120, max(1, self.winfo_screenwidth() - 80))
        height = min(740, max(1, self.winfo_screenheight() - 100))
        self.geometry(f"{width}x{height}")
        self.minsize(min(620, width), min(440, height))
        self.configure(background="#f3f6fa")
        self.transient(parent)

        heading = ttk.Frame(self, padding=(16, 12))
        heading.pack(fill=tk.X)
        ttk.Label(heading, text="CNN Model Architecture", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        count = sum(stage.parameters for stage in self.stages)
        ttk.Label(
            heading,
            text=f"{config.blocks} blocks | {config.blocks * config.convs_per_block} convolution layers | "
                 f"{count:,} parameters | Dropout {config.dropout:g}",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            heading, text="Configuration snapshot. Change the controls and click Show Model Arch again to refresh.",
            wraplength=max(200, width - 40),
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, background="#f8fafc", highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.bind("<Configure>", self._draw)
        self.canvas.bind("<MouseWheel>", self._scroll)
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=10)

    def _scroll(self, event):
        if event.delta:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _draw(self, event):
        canvas = self.canvas
        canvas.delete("all")
        columns = max(1, min(4, (event.width - 2 * self.MARGIN + self.GAP_X) // (self.CARD_WIDTH + self.GAP_X)))
        positions = []
        for index in range(len(self.stages)):
            row, column = divmod(index, columns)
            if row % 2:
                column = columns - 1 - column
            positions.append((
                self.MARGIN + column * (self.CARD_WIDTH + self.GAP_X),
                self.MARGIN + row * (self.CARD_HEIGHT + self.GAP_Y),
            ))
        for (x, y), (next_x, next_y) in zip(positions, positions[1:]):
            if y == next_y:
                rightward = next_x > x
                start = (x + self.CARD_WIDTH if rightward else x, y + self.CARD_HEIGHT / 2)
                end = (next_x if rightward else next_x + self.CARD_WIDTH, next_y + self.CARD_HEIGHT / 2)
            else:
                start = (x + self.CARD_WIDTH / 2, y + self.CARD_HEIGHT)
                end = (next_x + self.CARD_WIDTH / 2, next_y)
            canvas.create_line(*start, *end, fill="#526582", width=2, arrow=tk.LAST, arrowshape=(10, 12, 5))

        for stage, (x, y) in zip(self.stages, positions):
            color = (
                "#2379bd" if stage.title == "Input" else
                "#b86a21" if stage.title == "Output" else
                "#7951ad" if stage.title == "Classifier head" else "#23856c"
            )
            canvas.create_rectangle(x, y, x + self.CARD_WIDTH, y + self.CARD_HEIGHT, fill="white", outline=color, width=2)
            canvas.create_rectangle(x, y, x + self.CARD_WIDTH, y + 32, fill=color, outline=color)
            canvas.create_text(x + self.CARD_WIDTH / 2, y + 16, text=stage.title, fill="white", font=("Segoe UI", 11, "bold"))
            if len(stage.shape) == 3:
                for offset in (12, 6, 0):
                    canvas.create_rectangle(x + 72 + offset, y + 44 - offset / 2, x + 135 + offset, y + 72 - offset / 2, fill="#e3eef7", outline=color)
                start_y = y + 88
            else:
                start_y = y + 48
            for line in stage.details:
                canvas.create_text(
                    x + 12, start_y, text=line, anchor="nw", fill="#23334d",
                    font=("Segoe UI", 9), width=self.CARD_WIDTH - 24,
                )
                start_y += 28 if len(line) > 29 else 18
            shape = " x ".join(str(dimension) for dimension in stage.shape)
            canvas.create_text(
                x + 12, y + self.CARD_HEIGHT - 38, text=f"Output: {shape}",
                anchor="w", fill=color, font=("Segoe UI", 10, "bold"),
            )
            if stage.parameters:
                canvas.create_text(
                    x + 12, y + self.CARD_HEIGHT - 16, text=f"{stage.parameters:,} parameters",
                    anchor="w", fill="#61718a", font=("Segoe UI", 9),
                )
        rows = math.ceil(len(self.stages) / columns)
        canvas.configure(scrollregion=(
            0, 0,
            max(event.width, columns * (self.CARD_WIDTH + self.GAP_X) + self.MARGIN),
            rows * (self.CARD_HEIGHT + self.GAP_Y),
        ))
