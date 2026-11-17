import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer
from miditok import REMI


class TextConditioner(nn.Module):
    """
    Trích xuất Text Embedding từ Flan-T5 và chiếu (Project) về kích thước d_model của Transformer.
    """

    def __init__(self, t5_model_name="google/flan-t5-base", d_model=512, freeze=True):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5 = T5EncoderModel.from_pretrained(t5_model_name)

        if freeze:
            self.t5.eval()
            for param in self.t5.parameters():
                param.requires_grad = False

        t5_dim = self.t5.config.d_model
        self.projector = nn.Linear(t5_dim, d_model)

    def forward(self, text_ids, text_attention_mask):
        """
        Dùng lúc Train: nhận trực tiếp text_ids và text_attention_mask từ DataLoader.
        """
        with torch.no_grad():
            outputs = self.t5(input_ids=text_ids, attention_mask=text_attention_mask)

        text_embeds = self.projector(outputs.last_hidden_state)
        return text_embeds, text_attention_mask

    def encode_texts(self, text_list, device):
        """
        Dùng lúc Inference (generate): nhận danh sách câu prompt dạng string.
        """
        inputs = self.tokenizer(
            text_list, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        return self.forward(inputs.input_ids, inputs.attention_mask)


class TextToMidiTransformer(nn.Module):
    """
    Transformer Decoder chính tự sinh chuỗi MIDI Tokens (REMI) từ Text Condition.
    Tích hợp Classifier-Free Guidance (CFG) và xuất file .mid trực tiếp.
    """

    def __init__(
        self,
        midi_vocab_size,
        pad_token_id=0,
        d_model=512,
        nhead=8,
        num_layers=8,
        max_seq_len=1024,
        dropout=0.1,
        t5_model_name="google/flan-t5-base",
    ):
        super().__init__()
        self.midi_vocab_size = midi_vocab_size
        self.pad_token_id = pad_token_id
        self.d_model = d_model

        # 1. Text Encoder & Conditioning (Mặc định dùng flan-t5-base khớp với preprocess)
        self.text_conditioner = TextConditioner(t5_model_name=t5_model_name, d_model=d_model)

        # 2. MIDI Token Embeddings (0 -> midi_vocab_size-1: token nhạc, midi_vocab_size: BOS token)
        self.bos_token_id = midi_vocab_size
        self.token_embedding = nn.Embedding(
            midi_vocab_size + 1, d_model, padding_idx=pad_token_id
        )

        # Positional Encoding cho chuỗi sự kiện MIDI
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.emb_dropout = nn.Dropout(dropout)

        # 3. Transformer Decoder Backbone (norm_first=True giúp hội tụ ổn định hơn)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. Output Projection Head dự đoán token MIDI tiếp theo
        self.head = nn.Linear(d_model, midi_vocab_size)

    def _shift_right(self, midi_ids):
        """
        Teacher forcing chuẩn: [BOS, token_0, ..., token_(T-2)].
        """
        B, T = midi_ids.shape
        shifted_tokens = torch.full(
            (B, T), self.bos_token_id, dtype=torch.long, device=midi_ids.device
        )
        if T > 1:
            shifted_tokens[:, 1:] = midi_ids[:, :-1]
        return shifted_tokens

    def forward(self, midi_ids, text_ids, text_attention_mask, midi_padding_mask=None):
        """
        Huấn luyện Forward Pass:
            midi_ids: [B, T]
            text_ids: [B, L]
            text_attention_mask: [B, L] (1 = token thật, 0 = pad)
            midi_padding_mask: [B, T] (True = pad, False = token thật)
        """
        device = midi_ids.device
        B, T = midi_ids.shape

        # 1. Trích xuất Text Condition Embeddings
        text_embeds, text_mask = self.text_conditioner(text_ids, text_attention_mask)

        # 2. Dịch phải input MIDI để loại target leakage
        input_tokens = self._shift_right(midi_ids)

        # Dịch phải cả padding mask tương ứng (vị trí BOS đầu tiên luôn là False - không phải pad)
        if midi_padding_mask is not None:
            shifted_pad_mask = torch.zeros_like(midi_padding_mask, dtype=torch.bool)
            if T > 1:
                shifted_pad_mask[:, 1:] = midi_padding_mask[:, :-1]
        else:
            shifted_pad_mask = None

        # 3. Token Embedding + Positional Embedding
        midi_embeds = self.token_embedding(input_tokens) + self.pos_embedding[:, :T, :]
        midi_embeds = self.emb_dropout(midi_embeds)

        # 4. Masks cho Causal Attention & Cross Attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        memory_key_padding_mask = (text_mask == 0)

        # 5. Transformer Decoder
        output_embeds = self.transformer_decoder(
            tgt=midi_embeds,
            memory=text_embeds,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=shifted_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # 6. Predict Logits: [B, T, midi_vocab_size]
        logits = self.head(output_embeds)
        return logits

    @torch.no_grad()
    def generate(self, text_list, max_steps=512, temperature=1.0, top_k=50, cfg_scale=3.0):
        """
        Sinh chuỗi MIDI tokens từ danh sách câu mô tả văn bản (hỗ trợ Classifier-Free Guidance).
        """
        self.eval()
        device = next(self.parameters()).device
        B = len(text_list)

        def per_sample(value, cast=float):
            values = value if isinstance(value, (list, tuple)) else [value] * B
            if len(values) != B:
                raise ValueError(f"Sampling config must have {B} values, got {len(values)}")
            return [cast(item) for item in values]

        temperatures = torch.tensor(per_sample(temperature), device=device).unsqueeze(1).clamp_min(1e-5)
        cfg_scales = torch.tensor(per_sample(cfg_scale), device=device).unsqueeze(1)
        top_ks = per_sample(top_k, int)

        # Mã hóa đồng thời luồng có điều kiện (Cond) và không điều kiện (Uncond) cho CFG
        uncond_texts = [""] * B
        combined_text_embeds, combined_text_mask = self.text_conditioner.encode_texts(
            text_list + uncond_texts, device
        )
        memory_key_padding_mask = (combined_text_mask == 0)

        # Khởi tạo chuỗi bằng BOS token: [B, 1]
        generated_tokens = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        max_pos = self.pos_embedding.shape[1]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_steps):
            seq_len = generated_tokens.shape[1]
            if seq_len >= max_pos:
                break

            midi_embeds = self.token_embedding(generated_tokens) + self.pos_embedding[:, :seq_len, :]
            combined_midi_embeds = torch.cat([midi_embeds, midi_embeds], dim=0)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            output_embeds = self.transformer_decoder(
                tgt=combined_midi_embeds,
                memory=combined_text_embeds,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            # Lấy logits của bước thời gian cuối cùng: [2B, midi_vocab_size]
            next_logits = self.head(output_embeds[:, -1, :])

            # Áp dụng Classifier-Free Guidance (CFG)
            cond_logits, uncond_logits = next_logits.chunk(2, dim=0)
            guided_logits = uncond_logits + cfg_scales * (cond_logits - uncond_logits)
            guided_logits = guided_logits / temperatures

            # PAD/BOS/MASK are never musical events. EOS (id 2) remains available.
            guided_logits[:, [0, 1, 3]] = -float("Inf")

            # Top-K Sampling
            for row, k in enumerate(top_ks):
                if k > 0:
                    values, _ = torch.topk(guided_logits[row], min(k, guided_logits.size(-1)))
                    guided_logits[row] = guided_logits[row].masked_fill(guided_logits[row] < values[-1], -float("Inf"))

            probs = torch.softmax(guided_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]
            next_tokens = torch.where(finished.unsqueeze(1), torch.zeros_like(next_tokens), next_tokens)
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)
            finished |= next_tokens.squeeze(1).eq(2)
            if bool(finished.all()):
                break

        # Bỏ BOS token ở đầu và trả về [B, T]
        return generated_tokens[:, 1:]

    @staticmethod
    def save_tokens_to_midi(token_ids, tokenizer_config_path, output_paths):
        """
        Giải mã tensor token IDs [B, T] thành các file .mid hoàn chỉnh bằng MidiTok.
        """
        midi_tokenizer = REMI(params=tokenizer_config_path)
        token_list = token_ids.cpu().tolist()

        for ids, out_path in zip(token_list, output_paths):
            score = midi_tokenizer(ids)
            score.dump_midi(out_path)


# ==========================================
# Code kiểm thử Mô hình (Run Test)
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy test TextToMidiTransformer trên thiết bị: {device}")

    model = TextToMidiTransformer(
        midi_vocab_size=500,
        pad_token_id=0,
        d_model=512,
        nhead=8,
        num_layers=6,
        max_seq_len=1024,
    ).to(device)

    # Test Forward Pass (giống hệt lúc Train)
    dummy_midi = torch.randint(1, 500, (2, 128), device=device)
    dummy_text = torch.randint(1, 1000, (2, 20), device=device)
    dummy_mask = torch.ones((2, 20), dtype=torch.long, device=device)

    logits = model(dummy_midi, dummy_text, dummy_mask)
    print("Forward Logits Shape :", logits.shape)  # [2, 128, 500]

    # Test Inference (Generate)
    dummy_prompts = [
        "An emotional solo piano melody in C minor at 80 BPM",
        "Upbeat electronic pop track with synth and drums",
    ]
    gen_tokens = model.generate(dummy_prompts, max_steps=64, cfg_scale=3.0)
    print("Generated MIDI Shape :", gen_tokens.shape)  # [2, 64]
