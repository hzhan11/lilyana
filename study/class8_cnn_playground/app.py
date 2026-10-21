"""Interactive CIFAR-10 CNN experiments. Run with ``python app.py``."""

import math
from pathlib import Path
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

if __package__:
    from .model_architecture import ArchitectureWindow
    from .training import Config, parameter_count, run_training, validate_config
else:
    from model_architecture import ArchitectureWindow
    from training import Config, parameter_count, run_training, validate_config


class TrainingApp:
    """Own all Tk state on the main thread; workers communicate through events."""

    POLL_MS = 80
    MAX_LIVE_POINTS = 1200

    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.is_training = False
        self.output_dir = None
        self.history = []
        self.live_points = {"train": [], "val": []}
        self.control_widgets = []
        self._closing = False
        self._destroyed = False
        self._plot_dirty = False
        self._last_draw = 0.0
        self._estimate_after = None
        self._poll_after = None
        self._terminal_received = False
        self.architecture_window = None

        root.title("Class 8 | CNN Playground · CIFAR-10")
        available_width = max(1, root.winfo_screenwidth() - 40)
        available_height = max(1, root.winfo_screenheight() - 80)
        root.geometry(f"{min(1280, available_width)}x{min(850, available_height)}")
        root.minsize(min(1000, available_width), min(700, available_height))
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._configure_style()
        defaults = Config()
        integer_fields = {
            "blocks", "convs_per_block", "base_channels", "epochs",
            "batch_size", "train_limit", "val_limit", "seed", "cpu_threads",
        }
        self.variables = {}
        for name in (
            "blocks", "convs_per_block", "base_channels", "dropout", "epochs",
            "batch_size", "learning_rate", "optimizer", "weight_decay",
            "train_limit", "val_limit", "augment", "seed", "device", "cpu_threads",
        ):
            value = getattr(defaults, name)
            variable = (
                tk.BooleanVar(root, value=value) if name == "augment"
                else tk.StringVar(root, value=str(value))
            )
            self.variables[name] = variable
        self._integer_fields = integer_fields
        self.status_var = tk.StringVar(root, value="Ready — choose an experiment, then start training.")
        self.model_info_var = tk.StringVar(root, value="Calculating model size…")
        self.run_info_var = tk.StringVar(root, value="No run yet · CIFAR-10 · 10 classes · 32 × 32 RGB")
        self.progress_var = tk.StringVar(root, value="Waiting to start")
        self.preset_var = tk.StringVar(root, value="Quick experiment · 5,000 / 1,000")
        self._build_layout()
        for variable in self.variables.values():
            variable.trace_add("write", self._schedule_estimate)
        self.update_model_estimate()
        self._poll_after = root.after(self.POLL_MS, self._poll_events)

    def _configure_style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f3f6fa")
        style.configure("TLabel", background="#f3f6fa", foreground="#23334d")
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("Subtitle.TLabel", foreground="#61718a", font=("Segoe UI", 10))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabelframe", background="#f3f6fa")
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=9)
        style.configure("TButton", padding=6)
        style.configure("TCheckbutton", background="#f3f6fa")
        style.configure("Horizontal.TProgressbar", background="#2379bd")
        self.root.configure(background="#f3f6fa")

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="CNN Playground", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer, text="CLASS 8   /   Build a network. Train it. Compare training and validation.",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(1, 12))
        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, width=340)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_propagate(False)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.controls_canvas = tk.Canvas(
            left, background="#f3f6fa", highlightthickness=0, width=318,
        )
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=scrollbar.set)
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_frame = ttk.Frame(self.controls_canvas, padding=(0, 0, 10, 8))
        window = self.controls_canvas.create_window(
            (0, 0), window=self.controls_frame, anchor=tk.NW,
        )
        self.controls_frame.bind(
            "<Configure>",
            lambda event: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            ),
        )
        self.controls_canvas.bind(
            "<Configure>",
            lambda event: self.controls_canvas.itemconfigure(window, width=event.width),
        )
        self.root.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind("<Button-5>", self._on_mousewheel, add="+")
        self._build_controls()

        actions = ttk.Frame(left, padding=(0, 10, 12, 0))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        actions.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            actions, text="▶  Start training", style="Accent.TButton", command=self.start_training,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(
            actions, text="■  Stop", command=self.stop_training, state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=1, sticky="ew")

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(
            right, textvariable=self.run_info_var, style="Subtitle.TLabel", wraplength=750,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.figure = Figure(
            figsize=(8, 5.5), dpi=100, facecolor="#ffffff", layout="constrained",
        )
        self.loss_ax, self.accuracy_ax = self.figure.subplots(2, 1, sharex=True)
        self.lines = {}
        for name, axis, title, ylabel in (
            ("loss", self.loss_ax, "Loss  ·  lower is better", "Cross-entropy"),
            ("accuracy", self.accuracy_ax, "Accuracy  ·  higher is better", "Accuracy"),
        ):
            axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
            axis.set_ylabel(ylabel, fontsize=9)
            axis.set_facecolor("#fbfcfe")
            axis.grid(True, color="#e5ebf2", linewidth=.8)
            axis.spines[["top", "right"]].set_visible(False)
            for phase, color in (("train", "#2379bd"), ("val", "#e18a25")):
                line, = axis.plot(
                    [], [], color=color, linewidth=1.8,
                    label="Training" if phase == "train" else "Validation",
                )
                self.lines[(name, phase)] = line
            axis.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.accuracy_ax.set_xlabel("Epoch (live progress within each epoch)")
        self.accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._reset_charts()
        self.progress = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(10, 3))
        ttk.Label(right, textvariable=self.progress_var).grid(row=3, column=0, sticky="w")
        ttk.Label(
            right, textvariable=self.status_var, wraplength=740, font=("Segoe UI", 10, "bold"),
        ).grid(row=4, column=0, sticky="ew", pady=(5, 8))
        ttk.Label(right, text="EXPERIMENT LOG", style="Subtitle.TLabel").grid(
            row=5, column=0, sticky="w",
        )
        self.log_text = ScrolledText(
            right, height=6, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 9),
            background="#142238", foreground="#e6edf7", relief=tk.FLAT, padx=10, pady=8,
        )
        self.log_text.grid(row=6, column=0, sticky="ew", pady=(4, 0))
        self._log("Ready. The official 10,000-image test set is never used for training or validation.")

    def _section(self, title):
        frame = ttk.LabelFrame(
            self.controls_frame, text=title, style="Section.TLabelframe", padding=10,
        )
        frame.pack(fill=tk.X, pady=(0, 12))
        frame.columnconfigure(1, weight=1)
        return frame

    def _help(self, frame, text, row):
        ttk.Label(
            frame, text=text, wraplength=275, style="Subtitle.TLabel", justify=tk.LEFT,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(5, 3))

    def _field(self, frame, row, label, name, *, values=None, bounds=None, increment=1):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 8))
        if values is not None:
            widget = ttk.Combobox(
                frame, textvariable=self.variables[name], values=values,
                state="readonly", width=12,
            )
        else:
            widget = ttk.Spinbox(
                frame, textvariable=self.variables[name], from_=bounds[0], to=bounds[1],
                increment=increment, width=13,
            )
        widget.grid(row=row, column=1, sticky="ew", pady=5)
        self.control_widgets.append((widget, "readonly" if values is not None else "normal"))
        return widget

    def _build_controls(self):
        data = self._section("01  Dataset")
        self.preset_combo = ttk.Combobox(
            data, textvariable=self.preset_var, state="readonly", width=27,
            values=("Quick experiment · 5,000 / 1,000", "Full split · 45,000 / 5,000", "Custom limits"),
        )
        self.preset_combo.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        self.control_widgets.append((self.preset_combo, "readonly"))
        self._field(data, 1, "Training limit", "train_limit", bounds=(0, 45000), increment=1000)
        self._field(data, 2, "Validation limit", "val_limit", bounds=(0, 5000), increment=500)
        for name in ("train_limit", "val_limit"):
            self.variables[name].trace_add("write", self._sync_preset)
        self._help(
            data, "50,000 official training images → 45,000 train + 5,000 validation. "
            "A limit of 0 uses its full split. The separate 10,000 test images stay untouched.", 3,
        )
        augment = ttk.Checkbutton(
            data, text="Training augmentation (crop + flip)", variable=self.variables["augment"],
        )
        augment.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        self.control_widgets.append((augment, "normal"))

        model = self._section("02  Network architecture")
        self._field(model, 0, "Blocks", "blocks", bounds=(1, 4))
        self._field(model, 1, "Convs per block", "convs_per_block", bounds=(1, 3))
        self._field(model, 2, "Base channels", "base_channels", values=(8, 16, 32, 64, 128))
        self._field(model, 3, "Dropout", "dropout", bounds=(0, .8), increment=.05)
        self._help(
            model, "Depth = blocks × convolutions per block. Each block downsamples the image. "
            "Width = channels; later blocks double the base width. Dropout helps reduce overfitting.", 4,
        )
        ttk.Label(
            model, textvariable=self.model_info_var, wraplength=275, font=("Segoe UI", 10, "bold"),
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 2))
        self.architecture_button = ttk.Button(
            model, text="Show Model Arch", command=self.show_model_architecture,
        )
        self.architecture_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 2))

        train = self._section("03  Training settings")
        self._field(train, 0, "Epochs", "epochs", bounds=(1, 500))
        self._field(train, 1, "Batch size", "batch_size", bounds=(8, 512), increment=8)
        self._field(train, 2, "Learning rate", "learning_rate", bounds=(.000001, 1), increment=.0001)
        self._field(train, 3, "Optimizer", "optimizer", values=("AdamW", "SGD"))
        self._field(train, 4, "Weight decay", "weight_decay", bounds=(0, 1), increment=.0001)
        self._help(
            train, "One epoch is one pass over the training subset. Learning rate controls update size; "
            "weight decay discourages overly large weights.", 5,
        )

        runtime = self._section("04  Runtime & reproducibility")
        self._field(runtime, 0, "Device", "device", values=("auto", "cpu", "cuda"))
        self._field(runtime, 1, "CPU threads", "cpu_threads", bounds=(1, 32))
        self._field(runtime, 2, "Random seed", "seed", bounds=(0, 2**32 - 1))
        self._help(
            runtime, "Auto uses CUDA when available. CPU training is slower than CUDA. "
            "The first CIFAR-10 download is about 170 MB and requires internet access.", 3,
        )
        self._help(
            runtime, "Each run saves config.json, history.csv, best.pt, last.pt and a chart. "
            "Compare validation curves to spot overfitting.", 4,
        )

    def show_model_architecture(self):
        try:
            window = ArchitectureWindow(self.root, self.get_config())
        except Exception as exc:
            self._log(f"Cannot show model architecture: {exc}\n{traceback.format_exc()}")
            messagebox.showerror("Cannot show model architecture", str(exc), parent=self.root)
            return
        if self.architecture_window is not None and self.architecture_window.winfo_exists():
            self.architecture_window.destroy()
        self.architecture_window = window

    def _on_mousewheel(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget == self.controls_canvas:
                amount = -1 if getattr(event, "num", None) == 4 else 1
                if getattr(event, "delta", 0):
                    amount = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
                self.controls_canvas.yview_scroll(amount, "units")
                return "break"
            widget = getattr(widget, "master", None)
        return None

    def apply_preset(self, event=None):
        preset = self.preset_var.get()
        if preset.startswith("Quick"):
            sizes = (5000, 1000)
        elif preset.startswith("Full"):
            sizes = (0, 0)
        else:
            return
        for name, value in zip(("train_limit", "val_limit"), sizes):
            self.variables[name].set(str(value))

    def _sync_preset(self, *args):
        sizes = (self.variables["train_limit"].get(), self.variables["val_limit"].get())
        if sizes == ("5000", "1000"):
            self.preset_var.set("Quick experiment · 5,000 / 1,000")
        elif sizes in (("0", "0"), ("45000", "5000")):
            self.preset_var.set("Full split · 45,000 / 5,000")
        else:
            self.preset_var.set("Custom limits")

    def get_config(self):
        values = {}
        for name, variable in self.variables.items():
            raw = variable.get()
            try:
                if name == "augment":
                    values[name] = bool(raw)
                elif name in self._integer_fields:
                    values[name] = int(raw)
                elif name in ("dropout", "learning_rate", "weight_decay"):
                    values[name] = float(raw)
                    if not math.isfinite(values[name]):
                        raise ValueError("must be finite")
                else:
                    values[name] = raw
            except (ValueError, tk.TclError) as exc:
                raise ValueError(f"{name.replace('_', ' ').capitalize()}: enter a valid number.") from exc
        config = Config(**values)
        validate_config(config)
        return config

    def _schedule_estimate(self, *args):
        if self._destroyed:
            return
        if self._estimate_after is not None:
            self.root.after_cancel(self._estimate_after)
        self._estimate_after = self.root.after(300, self.update_model_estimate)

    def update_model_estimate(self):
        self._estimate_after = None
        try:
            config = self.get_config()
            count = parameter_count(config)
            channels = " → ".join(
                str(min(config.base_channels * 2**i, 512)) for i in range(config.blocks)
            )
            text = (
                f"{config.blocks * config.convs_per_block} convolution layers\n"
                f"{count:,} trainable parameters\nChannels: {channels}"
            )
            if count >= 8_000_000:
                text += "\nLarge model: CPU training may be very slow."
            self.model_info_var.set(text)
        except ValueError as exc:
            self.model_info_var.set(f"Check settings: {exc}")
        except Exception as exc:
            self.model_info_var.set(f"Model estimate failed: {exc}")
            self._log(traceback.format_exc())

    def _set_active(self, active):
        self.is_training = active
        for widget, normal_state in self.control_widgets:
            widget.configure(state=tk.DISABLED if active or self._closing else normal_state)
        self.start_button.configure(state=tk.DISABLED if active or self._closing else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if active and not self._closing else tk.DISABLED)

    def start_training(self):
        if self._closing or self.is_training or (self.worker is not None and self.worker.is_alive()):
            return
        try:
            config = self.get_config()
            count = parameter_count(config)
        except Exception as exc:
            self._log(f"Invalid configuration: {exc}")
            messagebox.showerror("Cannot start training", str(exc), parent=self.root)
            return
        if count >= 8_000_000 and config.device != "cuda":
            if not messagebox.askyesno(
                "Large CNN", f"This network has {count:,} parameters. CPU training may be very slow "
                "and use substantial memory.\n\nContinue with these settings?", parent=self.root,
            ):
                return
        self.stop_event = threading.Event()
        self.output_dir = None
        self._terminal_received = False
        self._set_active(True)
        self._reset_charts()
        self.progress.configure(value=0)
        self.progress_var.set("Preparing data and model…")
        self.status_var.set("Preparing CIFAR-10. First download ≈170 MB; CPU is slower than CUDA.")
        self.run_info_var.set(
            f"{config.blocks} blocks × {config.convs_per_block} convs · "
            f"base width {config.base_channels} · {count:,} parameters · {config.optimizer}"
        )
        self._log(
            f"\nStarting experiment: {config.blocks * config.convs_per_block} convolution layers, "
            f"{count:,} parameters; {config.epochs} epochs, batch {config.batch_size}, "
            f"learning rate {config.learning_rate:g}."
        )
        self.worker = threading.Thread(
            target=self._run_worker, args=(config, self.events, self.stop_event),
            name="cifar10-training", daemon=False,
        )
        try:
            self.worker.start()
        except Exception as exc:
            self._set_active(False)
            self._log(traceback.format_exc())
            messagebox.showerror("Could not start worker", str(exc), parent=self.root)

    @staticmethod
    def _run_worker(config, events, stop_event):
        try:
            run_training(config, events.put, stop_event)
        except Exception as exc:
            events.put({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})

    def stop_training(self):
        if not self.is_training:
            return
        self.stop_event.set()
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopping safely… waiting for the current operation to finish.")
        self._log("Stop requested. Downloads or the current batch may need time to finish.")

    def _reset_charts(self):
        self.history.clear()
        self.live_points = {"train": [], "val": []}
        for line in self.lines.values():
            line.set_data([], [])
        self.loss_ax.set_xlim(0, 1)
        self.loss_ax.set_ylim(0, 3)
        self.accuracy_ax.set_ylim(0, 1)
        self.canvas.draw_idle()
        self._plot_dirty = False

    def _log(self, text):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')}  {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_events(self):
        self._poll_after = None
        if self._destroyed:
            return
        for _ in range(250):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                self.handle_event(event)
            except Exception as exc:
                self._log(traceback.format_exc())
                self.stop_event.set()
                self.status_var.set(f"Display error: {exc}. Training stop requested.")
                messagebox.showerror("Training display error", str(exc), parent=self.root)
        if self._plot_dirty and time.monotonic() - self._last_draw >= .2:
            self._draw_charts()
        alive = self.worker is not None and self.worker.is_alive()
        if self.is_training and not alive and self.events.empty():
            if not self._terminal_received:
                self.status_var.set("Training ended unexpectedly without a completion event.")
                self._log(self.status_var.get())
                messagebox.showerror("Training ended unexpectedly", self.status_var.get(), parent=self.root)
            self._set_active(False)
        if self._closing and not alive and self.events.empty():
            self._destroy()
            return
        self._poll_after = self.root.after(self.POLL_MS, self._poll_events)

    def handle_event(self, event):
        """Handle an engine event on the Tk thread (also useful for UI tests)."""
        kind = event.get("type")
        if kind == "status":
            text = event.get("message", "")
            if not self.stop_event.is_set():
                self.status_var.set(text)
            self._log(text)
        elif kind == "started":
            self.output_dir = event.get("output_dir") or None
            self._reset_charts()
            device = event.get("device", "unknown")
            self.run_info_var.set(
                f"Device: {device}  ·  {event.get('parameters', 0):,} parameters  ·  "
                f"{event.get('train_size', 0):,} train / {event.get('val_size', 0):,} validation"
            )
            self._log(self.run_info_var.get())
            if self.output_dir:
                self._log(f"Run folder: {self.output_dir}")
        elif kind == "metrics":
            phase = event.get("phase")
            if phase not in self.live_points:
                return
            point = (float(event["epoch"]), float(event["loss"]), float(event["accuracy"]))
            points = self.live_points[phase]
            points.append(point)
            if len(points) > self.MAX_LIVE_POINTS:
                points[:] = points[::2] + ([points[-1]] if len(points) % 2 == 0 else [])
            self._plot_dirty = True
            epoch_number = event.get("epoch_number", 1)
            epochs = max(1, event.get("epochs", 1))
            batch, batches = event.get("batch", 0), max(1, event.get("batches", 1))
            phase_fraction = min(1, max(0, batch / batches))
            fraction = (epoch_number - 1 + (.5 * phase_fraction if phase == "train"
                        else .5 + .5 * phase_fraction)) / epochs
            self.progress.configure(value=min(100, max(0, 100 * fraction)))
            self.progress_var.set(
                f"Epoch {epoch_number}/{epochs} · {'Training' if phase == 'train' else 'Validation'} "
                f"· batch {batch}/{batches} · loss {point[1]:.4f} · accuracy {point[2]:.1%}"
            )
            if not self.stop_event.is_set():
                self.status_var.set("Training… live curves update as batches finish.")
        elif kind == "epoch":
            self.history.append(dict(event))
            self._plot_dirty = True
            self._log(
                f"Epoch {event['epoch']:>3} | train loss {event['train_loss']:.4f}, "
                f"acc {event['train_accuracy']:.1%} | val loss {event['val_loss']:.4f}, "
                f"acc {event['val_accuracy']:.1%} | {event.get('elapsed_seconds', 0):.1f}s"
            )
        elif kind in ("done", "stopped"):
            self._terminal_received = True
            self.output_dir = event.get("output_dir") or self.output_dir
            best = event.get("best_accuracy")
            best_text = f" · best validation accuracy {best:.1%}" if best is not None else ""
            text = (
                f"{'Completed' if kind == 'done' else 'Stopped'} · "
                f"{event.get('epochs_completed', len(self.history))} full epochs{best_text}"
            )
            self.status_var.set(text)
            self.progress_var.set(text)
            self.stop_button.configure(state=tk.DISABLED)
            if kind == "done":
                self.progress.configure(value=100)
            self._log(text)
            self._draw_charts()
            self._save_chart()
        elif kind == "error":
            self._terminal_received = True
            self.stop_event.set()
            message = event.get("message", "Unknown training error")
            self.status_var.set(f"Training failed: {message}")
            self.progress_var.set("Training failed — see the experiment log.")
            self.stop_button.configure(state=tk.DISABLED)
            self._log(f"ERROR: {message}\n{event.get('traceback', '')}")
            messagebox.showerror("Training failed", message, parent=self.root)

    def _draw_charts(self):
        max_epoch = 1.0
        for phase in ("train", "val"):
            points = {point[0]: point[1:] for point in self.live_points[phase]}
            # Retain every full-epoch result even when live batch points are sampled.
            for record in self.history:
                points[float(record["epoch"])] = (
                    record[f"{phase}_loss"], record[f"{phase}_accuracy"],
                )
            ordered = sorted(points.items())
            x = [epoch for epoch, _ in ordered]
            if x:
                max_epoch = max(max_epoch, x[-1])
            for index, metric in enumerate(("loss", "accuracy")):
                self.lines[(metric, phase)].set_data(x, [values[index] for _, values in ordered])
        self.loss_ax.relim()
        self.loss_ax.autoscale_view(scalex=False, scaley=True)
        self.loss_ax.set_ylim(bottom=0)
        self.loss_ax.set_xlim(0, max_epoch * 1.025)
        self.accuracy_ax.set_ylim(0, 1)
        self.canvas.draw_idle()
        self._last_draw = time.monotonic()
        self._plot_dirty = False

    def _save_chart(self):
        if not self.output_dir or not (self.history or any(self.live_points.values())):
            return
        path = Path(self.output_dir) / "training_curves.png"
        try:
            self.figure.savefig(path, dpi=150, facecolor=self.figure.get_facecolor())
        except Exception as exc:
            self._log(f"Training result retained, but chart export failed: {path}\n{traceback.format_exc()}")
            self.status_var.set(self.status_var.get() + " · chart export failed")
            messagebox.showerror(
                "Chart could not be saved",
                f"The training result is retained, but the chart could not be saved:\n{path}\n\n{exc}",
                parent=self.root,
            )
        else:
            self._log(f"Chart saved: {path}")

    def on_close(self):
        if self._closing:
            return
        alive = self.worker is not None and self.worker.is_alive()
        if alive or self.is_training:
            if not messagebox.askyesno(
                "Stop training and close?",
                "Training is active. Stop safely and close when the current operation finishes?\n\n"
                "An in-progress download may take time. The window will remain responsive.",
                parent=self.root,
            ):
                return
            self._closing = True
            self.stop_training()
            self._set_active(True)
            self.status_var.set("Closing after training stops safely. Please wait…")
        else:
            self._destroy()

    def _destroy(self):
        self._destroyed = True
        for callback in (self._estimate_after, self._poll_after):
            if callback is not None:
                self.root.after_cancel(callback)
        # TkAgg owns an idle callback separate from our event polling.
        idle_draw = getattr(self.canvas, "_idle_draw_id", None)
        if idle_draw is not None:
            self.canvas.get_tk_widget().after_cancel(idle_draw)
            self.canvas._idle_draw_id = None
        self.figure.clear()
        self.root.destroy()


def main():
    root = tk.Tk()
    TrainingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
