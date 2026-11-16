# Cài đặt thư viện nếu chạy lần đầu trên Colab:
# !pip install -q miditok symusic transformers datasets torch tqdm

import os
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
from symusic import Score
from miditok import REMI, TokenizerConfig
from transformers import T5Tokenizer

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ==========================================
MAX_MIDI_TOKENS = 1024
MAX_TEXT_TOKENS = 128
SUBSET_SIZE = 10000  # None = xử lý toàn bộ ~168k mẫu

# Tự động dò tìm thư mục midi_files ở /content hoặc Google Drive
possible_dirs = [
    Path("/content/MusicGen/data_midicaps/midi_files"),
    Path("/content/drive/MyDrive/MusicGen/data_midicaps/midi_files"),
    Path("./MusicGen/data_midicaps/midi_files"),
    Path("./data_midicaps/midi_files"),
]

MIDI_EXTRACT_DIR = next((d for d in possible_dirs if d.exists()), None)
if MIDI_EXTRACT_DIR is None:
    raise FileNotFoundError(
        "❌ Không tìm thấy thư mục 'midi_files'! Hãy kiểm tra lại xem bạn đã mount Google Drive chưa."
    )

# Thư mục lưu file .pt sau khi xử lý (ưu tiên lưu vào Google Drive nếu đã mount)
if Path("/content/drive/MyDrive").exists():
    OUTPUT_DIR = Path("/content/drive/MyDrive/MusicGen/processed_data")
else:
    OUTPUT_DIR = Path("/content/MusicGen/processed_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Thư mục MIDI đầu vào : {MIDI_EXTRACT_DIR}")
print(f"💾 Thư mục lưu file .pt  : {OUTPUT_DIR}")

# ==========================================
# 2. KHỞI TẠO TOKENIZER & TÁI TẠO DANH SÁCH FILE
# ==========================================
midi_config = TokenizerConfig(
    num_velocities=16,
    use_chords=True,
    use_tempos=True,
    use_programs=True,
    one_token_stream_for_programs=True,
)
midi_tokenizer = REMI(midi_config)
tokenizer_save_path = OUTPUT_DIR / "midi_tokenizer_config.json"
midi_tokenizer.save(tokenizer_save_path)

text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

print("⏳ Đang nạp metadata MidiCaps từ Hugging Face...")
dataset = load_dataset("amaai-lab/MidiCaps", split="train")

valid_pairs = []
for item in tqdm(dataset, desc="Đang khớp file MIDI"):
    full_midi_path = MIDI_EXTRACT_DIR / item["location"]
    if full_midi_path.exists():
        valid_pairs.append((str(full_midi_path), item["caption"]))

if SUBSET_SIZE is not None:
    valid_pairs = valid_pairs[:SUBSET_SIZE]

print(f"✅ Tìm thấy {len(valid_pairs)} cặp (MIDI, Caption) hợp lệ.")

# ==========================================
# 3. HÀM XỬ LÝ ĐA LUỒNG (WORKER)
# ==========================================
# Khởi tạo biến toàn cục cho từng worker process để tránh nạp lại tokenizer nhiều lần
_worker_midi_tok = None
_worker_text_tok = None

def _init_worker(tok_path):
    global _worker_midi_tok, _worker_text_tok
    _worker_midi_tok = REMI(params=tok_path)
    _worker_text_tok = T5Tokenizer.from_pretrained("google/flan-t5-base")

def _process_single_item(pair):
    midi_path, caption = pair
    try:
        score = Score(midi_path)
        midi_tokens = _worker_midi_tok(score).ids

        if len(midi_tokens) < 32:
            return None

        midi_ids = torch.tensor(midi_tokens[:MAX_MIDI_TOKENS], dtype=torch.int32)

        text_enc = _worker_text_tok(
            caption,
            max_length=MAX_TEXT_TOKENS,
            truncation=True,
            return_tensors="pt"
        )
        text_ids = text_enc.input_ids.squeeze(0).to(torch.int32)

        return {
            "text_ids": text_ids,
            "midi_ids": midi_ids,
        }
    except Exception:
        return None

# ==========================================
# 4. THỰC THI VÀ LƯU FILE .PT
# ==========================================
if __name__ == "__main__":
    num_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"🚀 Bắt đầu tokenize toàn bộ dữ liệu với {num_workers} luồng CPU...")

    processed_dataset = []
    skipped = 0

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_save_path,)
    ) as executor:
        results = tqdm(
            executor.map(_process_single_item, valid_pairs, chunksize=64),
            total=len(valid_pairs),
            desc="Tokenizing"
        )
        for res in results:
            if res is not None:
                processed_dataset.append(res)
            else:
                skipped += 1

    save_path = OUTPUT_DIR / f"midicaps_processed_{len(processed_dataset)}.pt"
    torch.save({
        "data": processed_dataset,
        "midi_vocab_size": len(midi_tokenizer),
        "pad_token_id": midi_tokenizer["PAD_None"],
    }, save_path)

    print(f"\n🎉 HOÀN TẤT!")
    print(f"📦 Số mẫu hợp lệ đã lưu : {len(processed_dataset)}")
    print(f"⚠️ Số file lỗi/quá ngắn : {skipped}")
    print(f"📊 Kích thước Vocab MIDI : {len(midi_tokenizer)}")
    print(f"💾 Đường dẫn file .pt    : {save_path}")