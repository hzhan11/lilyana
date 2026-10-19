"""Draw the network architecture configured in train_model.py."""

import ast
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DAY5 = Path(__file__).resolve().parent
FONT_PATH = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc"
OUTPUT_PATH = DAY5 / "network_structure.png"


def main():
    settings = {}
    tree = ast.parse((DAY5 / "train_model.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("HIDDEN_SIZE", "NUM_CLASSES"):
                    settings[target.id] = ast.literal_eval(node.value)
    hidden_size, outputs = settings["HIDDEN_SIZE"], settings["NUM_CLASSES"]
    if not isinstance(hidden_size, int) or hidden_size <= 0 or outputs != 10:
        raise ValueError("Expected a positive HIDDEN_SIZE and 10 digit outputs.")
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")

    image = Image.new("RGB", (2000, 1360), "#F4F7FB")
    draw = ImageDraw.Draw(image)

    def text(x, y, content, size=28, color="#172B4D", centered=False):
        font = ImageFont.truetype(str(FONT_PATH), size)
        if centered:
            x -= draw.textlength(content, font=font) / 2
        bounds = draw.textbbox((x, y), content, font=font, anchor="lt")
        if not (0 <= bounds[0] < bounds[2] <= image.width and 0 <= bounds[1] < bounds[3] <= image.height):
            raise ValueError(f"Text exceeds image bounds: {content}")
        draw.text((x, y), content, font=font, fill=color, anchor="lt")

    def panel(bounds, fill="white"):
        draw.rounded_rectangle(bounds, radius=24, fill=fill, outline="#D9E2F0", width=2)

    def arrow(start, end):
        draw.line((*start, *end), fill="#2864DC", width=4)
        draw.polygon((end, (end[0] - 16, end[1] - 9), (end[0] - 16, end[1] + 9)), fill="#2864DC")

    def nodes(x, count, prefix, color, fill):
        indices = list(range(1, count + 1)) if count <= 6 else [1, 2, 3, None, count - 2, count - 1, count]
        if prefix == "":
            indices = list(range(count))
        points = []
        for index, number in enumerate(indices):
            y = 450 + (450 * index / (len(indices) - 1) if len(indices) > 1 else 225)
            if number is None:
                for offset in (-12, 0, 12):
                    draw.ellipse((x - 3, y + offset - 3, x + 3, y + offset + 3), fill="#526582")
                continue
            points.append((x, y))
            radius = 22 if prefix == "" else 30
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=color, width=3)
            text(x, y - 13, f"{prefix}{number}", 17 if prefix else 24, color, centered=True)
        return points

    # Step 1: Draw the three layers and parameter dimensions.
    text(64, 34, "DAY 5 / NEURAL NETWORK", 23, "#2864DC")
    text(64, 90, f"神经网络结构：4096 → {hidden_size} → {outputs}", 48)
    text(64, 166, f"HIDDEN_SIZE = {hidden_size}；两层可训练的全连接层，输入层不计入可训练层数。", 28, "#526582")
    draw.line((64, 225, 1936, 225), fill="#D9E2F0", width=2)
    for bounds in ((64, 270, 550, 1030), (760, 270, 1240, 1030), (1450, 270, 1936, 1030)):
        panel(bounds)
    text(305, 300, "输入层", 36, "#2864DC", centered=True)
    text(305, 355, "4096 个像素值", 28, centered=True)
    text(1000, 300, "隐藏层", 36, "#168568", centered=True)
    text(1000, 355, f"{hidden_size} 个神经元 + ReLU", 28, centered=True)
    text(1693, 300, "输出层", 36, "#7951BA", centered=True)
    text(1693, 355, "10 个分数，对应 0～9", 27, centered=True)
    for x, lines in (
        (650, ["第一层", f"W1: 4096 × {hidden_size}", f"b1: {hidden_size}"]),
        (1345, ["第二层", f"W2: {hidden_size} × 10", "b2: 10"]),
    ):
        for row, line in enumerate(lines):
            text(x, 285 + row * 38, line, 21, "#526582", centered=True)

    # Step 2: Connect representative neurons, then redraw nodes above the lines.
    input_points = nodes(380, 4096, "x", "#2864DC", "#EAF1FF")
    hidden_points = nodes(1000, hidden_size, "h", "#168568", "#E7F5EF")
    output_points = nodes(1580, outputs, "", "#7951BA", "#F0EAF9")
    for left, right in ((input_points, hidden_points), (hidden_points, output_points)):
        for start in left:
            for end in right:
                end_offset = 24 if right is output_points else 32
                draw.line((start[0] + 32, start[1], end[0] - end_offset, end[1]), fill="#C7D5E9", width=2)
    nodes(380, 4096, "x", "#2864DC", "#EAF1FF")
    nodes(1000, hidden_size, "h", "#168568", "#E7F5EF")
    nodes(1580, outputs, "", "#7951BA", "#F0EAF9")
    for digit, (_, y) in enumerate(output_points):
        text(1640, y - 12, f"数字 {digit} 的分数", 23, "#7951BA")
    text(305, 957, "64×64 灰度图 → 归一化 → 展开", 23, centered=True)
    text(1000, 957, "ReLU：负数变成 0", 25, "#168568", centered=True)
    text(1693, 957, "这里的输出是分数，还不是概率", 23, centered=True)

    # Step 3: Explain the output and the omitted neurons.
    panel((64, 1070, 1936, 1245), "#EAF1FF")
    for x, content in ((280, "10 个输出分数"), (760, "Softmax"), (1240, "10 个预测概率"), (1720, "选最大值 → 预测数字")):
        text(x, 1095, content, 28, "#2864DC", centered=True)
    for left, right in ((450, 620), (900, 1080), (1420, 1500)):
        arrow((left, 1112), (right, 1112))
    text(1000, 1170, "第一层：h = ReLU(x @ W1 + b1)     |     第二层：scores = h @ W2 + b2", 27, centered=True)
    text(1000, 1280, "仅显示部分输入 / 隐藏节点，省略号代表未画出的节点；实际相邻两层之间是全连接。", 27, "#526582", centered=True)
    image.save(OUTPUT_PATH, format="PNG")
    print(f"Network diagram saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
