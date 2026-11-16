import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class MidiCapsDataset(Dataset):
    """
    Dataset nạp dữ liệu Text-to-MIDI từ các file chunk .pt đã tiền xử lý.
    Tự động gộp nhiều file (midicaps_0_10000.pt, midicaps_10000_30000.pt...)
    và chia tập training / validation nhất quán.
    """

    def __init__(
        self,
        token_dir="/content/drive/MyDrive/MusicGen/processed_data",
        split="training",
        val_ratio=0.1,
        seed=42,
    ):
        pt_files = sorted(glob.glob(os.path.join(token_dir, "midicaps_*.pt")))
        if not pt_files:
            raise FileNotFoundError(
                f"❌ Không tìm thấy file midicaps_*.pt nào trong '{token_dir}'. Hãy chạy preprocess trước!"
            )

        all_samples = []
        self.midi_vocab_size = None
        self.midi_pad_id = 0

        for file_path in pt_files:
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
            all_samples.extend(checkpoint["data"])
            self.midi_vocab_size = checkpoint["midi_vocab_size"]
            self.midi_pad_id = checkpoint["pad_token_id"]

        # Xáo trộn cố định bằng seed để tập train và val không bao giờ bị lẫn lộn
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1.0 - val_ratio))
        if split == "training":
            self.samples = all_samples[:split_idx]
        elif split == "validation":
            self.samples = all_samples[split_idx:]
        else:
            self.samples = all_samples

        print(
            f"✅ [{split.upper()}] Đã nạp {len(self.samples)}/{len(all_samples)} mẫu "
            f"từ {len(pt_files)} file .pt (MIDI Vocab: {self.midi_vocab_size}, Pad ID: {self.midi_pad_id})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            # Chuyển từ int32 sang long (int64) để dùng cho nn.Embedding
            "text_ids": item["text_ids"].long(),
            "midi_ids": item["midi_ids"].long(),
        }


def create_collate_fn(text_pad_id=0, midi_pad_id=0):
    """
    Hàm gom batch và thực hiện Dynamic Padding (đệm độ dài bằng nhau trong từng batch).
    """
    def collate_fn(batch):
        text_list = [item["text_ids"] for item in batch]
        midi_list = [item["midi_ids"] for item in batch]

        # Pad chuỗi văn bản về cùng độ dài lớn nhất trong batch: [Batch, Max_Text_Len]
        padded_text = pad_sequence(text_list, batch_first=True, padding_value=text_pad_id)
        text_attention_mask = (padded_text != text_pad_id).long()

        # Pad chuỗi MIDI về cùng độ dài lớn nhất trong batch: [Batch, Max_Midi_Len]
        padded_midi = pad_sequence(midi_list, batch_first=True, padding_value=midi_pad_id)
        # Mask vị trí thực của MIDI (True = token thật, False = vị trí padding)
        midi_padding_mask = (padded_midi == midi_pad_id)

        return {
            "text_ids": padded_text,
            "text_attention_mask": text_attention_mask,
            "midi_ids": padded_midi,
            "midi_padding_mask": midi_padding_mask,
        }

    return collate_fn


def get_dataloaders(
    token_dir="/content/drive/MyDrive/MusicGen/processed_data",
    batch_size=16,
    num_workers=2,
    val_ratio=0.1,
):
    train_dataset = MidiCapsDataset(token_dir=token_dir, split="training", val_ratio=val_ratio)
    val_dataset = MidiCapsDataset(token_dir=token_dir, split="validation", val_ratio=val_ratio)

    # T5Tokenizer luôn có pad_token_id = 0
    collate_fn = create_collate_fn(text_pad_id=0, midi_pad_id=train_dataset.midi_pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.midi_vocab_size, train_dataset.midi_pad_id


# ==========================================
# Code kiểm thử Dataloader
# ==========================================
if __name__ == "__main__":
    TOKEN_DIR = "/content/drive/MyDrive/MusicGen/processed_data"
    if not os.path.exists(TOKEN_DIR):
        TOKEN_DIR = "./data/processed_tokens"

    if os.path.exists(TOKEN_DIR):
        train_loader, val_loader, vocab_size, pad_id = get_dataloaders(
            token_dir=TOKEN_DIR,
            batch_size=8,
            num_workers=0,
        )

        for batch in train_loader:
            print("\n--- KẾT QUẢ LOAD 1 BATCH TRAIN ---")
            print("Text IDs Shape       :", batch["text_ids"].shape)
            print("Text Mask Shape      :", batch["text_attention_mask"].shape)
            print("MIDI IDs Shape       :", batch["midi_ids"].shape)
            print("MIDI Pad Mask Shape  :", batch["midi_padding_mask"].shape)
            break
    else:
        print(f"Chưa tìm thấy thư mục {TOKEN_DIR}. Hãy chạy file preprocess trước!")