# Cài đặt thư viện cần thiết (chạy trên terminal hoặc cell Colab)
# !pip install -q datasets huggingface_hub tqdm

import os
import tarfile
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# 1. Cấu hình thư mục lưu trữ (khuyên dùng thư mục trên Google Drive nếu chạy Colab)
BASE_DIR = Path("./data_midicaps")  # Đổi thành Path("/content/drive/MyDrive/MusicGen/data_midicaps") nếu lưu Drive
MIDI_EXTRACT_DIR = BASE_DIR / "midi_files"
BASE_DIR.mkdir(parents=True, exist_ok=True)
MIDI_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "amaai-lab/MidiCaps"

# 2. Tải phần Metadata & Captions từ Hugging Face Datasets
print("⏳ Đang tải metadata và text captions của MidiCaps...")
dataset = load_dataset(REPO_ID, split="train")
print(f"✅ Đã tải {len(dataset)} mẫu metadata.")
print("Ví dụ 1 mẫu metadata:", dataset[0])

# 3. Tải tệp nén chứa toàn bộ file MIDI (~1.2 GB) từ Hugging Face Hub
print("\n⏳ Đang tải kho file MIDI (midicaps.tar.gz)...")
tar_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="midicaps.tar.gz",
    repo_type="dataset",
    local_dir=BASE_DIR,
)

# 4. Giải nén tệp tar.gz
print(f"\n⏳ Đang giải nén vào {MIDI_EXTRACT_DIR}...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=MIDI_EXTRACT_DIR)
print("✅ Giải nén hoàn tất!")

# 5. Kiểm tra và lọc ra danh sách cặp (caption, đường dẫn file .mid hợp lệ)
valid_pairs = []
for item in tqdm(dataset, desc="Đang kiểm tra file MIDI"):
    # Cột 'location' chứa đường dẫn tương đối tới file .mid
    midi_rel_path = item["location"]
    full_midi_path = MIDI_EXTRACT_DIR / midi_rel_path

    if full_midi_path.exists():
        valid_pairs.append({
            "midi_path": str(full_midi_path),
            "caption": item["caption"],
            "genre": item.get("genre", []),
            "mood": item.get("mood", []),
            "tempo": item.get("tempo", 120),
            "key": item.get("key", ""),
        })

print(f"\n🎉 Sẵn sàng huấn luyện với {len(valid_pairs)} cặp (Text, MIDI) hợp lệ!")
print("Mẫu đầu tiên:", valid_pairs[0])