import random

# Định nghĩa các thành phần
subjects = ["", "xe ", "robot ", "bạn "]
actions = {
    "MOVE_F": ["đi thẳng", "tiến lên", "tiến về phía trước", "đi tới", "tiến"],
    "TURN_L": ["rẽ trái", "quẹo trái", "sang trái", "bẻ lái sang trái", "trái"],
    "TURN_R": ["rẽ phải", "quẹo phải", "sang phải", "bẻ lái sang phải", "phải"],
    "STOP": ["dừng lại", "đứng yên", "dừng", "ngắt động cơ"],
    "MOVE_B": ["lùi lại", "lùi", "về sau", "đi ngược lại"]
}
connectors = [" rồi ", " sau đó ", " tiếp theo là ", " và "]
suffixes = ["", " đi", " ngay", " nhé"]

dataset = []

# Tạo câu đơn
for cmd, phrases in actions.items():
    for p in phrases:
        for s in subjects:
            for suf in suffixes:
                sentence = f"{s}{p}{suf}".strip()
                dataset.append(f"{sentence} <SEP> {cmd} <EOS>")

# Tạo câu kép (Ví dụ: đi thẳng rồi rẽ trái)
for p1 in actions["MOVE_F"]:
    for cmd2 in ["TURN_L", "TURN_R", "STOP"]:
        for p2 in actions[cmd2]:
            for conn in connectors:
                sentence = f"{p1}{conn}{p2}".strip()
                dataset.append(f"{sentence} <SEP> MOVE_F {cmd2} <EOS>")

# Lưu ra file
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(list(set(dataset)))) # Xóa trùng lặp