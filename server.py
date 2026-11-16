"""MusicGen2.0 API: text prompt -> MIDI -> lightweight WAV preview."""
from __future__ import annotations

import json, math, os, subprocess, threading, time, uuid, wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from model import TextToMidiTransformer

ROOT = Path(__file__).resolve().parent
STATIC_DIR, MODEL_DIR, DATA_DIR = ROOT / "static", ROOT / "models", ROOT / "data"
OUTPUT_DIR, HISTORY_FILE = DATA_DIR / "generated", DATA_DIR / "history.json"
CHECKPOINT = MODEL_DIR / "best_text2midi_model.pt"
TOKENIZER_CONFIG = MODEL_DIR / "midi_tokenizer_config.json"
SOUNDFONT_DIR = ROOT / "soundfonts"
SOUNDFONT = SOUNDFONT_DIR / "FluidR3 GM.sf2"
FLUIDSYNTH = next(SOUNDFONT_DIR.glob("fluidsynth-*-win10-x64-cpp11/bin/fluidsynth.exe"), None)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    session_id: str | None = None
    max_steps: int = Field(default=int(os.getenv("MUSICGEN_MAX_STEPS", "256")), ge=32, le=512)
    temperature: float = Field(default=float(os.getenv("MUSICGEN_TEMPERATURE", "1.0")), ge=0.1, le=2.0)
    top_k: int = Field(default=int(os.getenv("MUSICGEN_TOP_K", "50")), ge=0, le=200)
    cfg_scale: float = Field(default=float(os.getenv("MUSICGEN_CFG_SCALE", "3.0")), ge=0.0, le=8.0)

class ModelService:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.loaded_at = None

    def load(self):
        if self.model is not None:
            return self.model
        if not CHECKPOINT.exists() or not TOKENIZER_CONFIG.exists():
            raise RuntimeError("Thiếu checkpoint hoặc cấu hình MIDI tokenizer")
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model = TextToMidiTransformer(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        if self.device.type == "cuda":
            model.half()  # phù hợp VRAM 4 GB
            torch.backends.cuda.matmul.allow_tf32 = True
        self.model = model.to(self.device).eval()
        self.loaded_at = time.time()
        return self.model

    def generate(self, request, midi_path):
        with self.lock, torch.inference_mode():
            model = self.load()
            tokens = model.generate([request.prompt], max_steps=request.max_steps,
                                    temperature=request.temperature, top_k=request.top_k,
                                    cfg_scale=request.cfg_scale)
            model.save_tokens_to_midi(tokens, str(TOKENIZER_CONFIG), [str(midi_path)])
        from symusic import Score
        return Score(str(midi_path))

service = ModelService()
app = FastAPI(title="MusicGen2.0", version="0.1.0")

def now(): return datetime.now(timezone.utc).isoformat()

def read_history():
    if not HISTORY_FILE.exists(): return []
    try: return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError): return []

def write_history(items):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp = HISTORY_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(HISTORY_FILE)

def score_notes_in_seconds(score: Any):
    """Đổi note MIDI từ tick sang giây theo toàn bộ tempo map."""
    ticks_per_quarter = max(1, int(score.ticks_per_quarter))
    tempos = sorted(score.tempos, key=lambda tempo: tempo.time)
    if not tempos or tempos[0].time > 0:
        from symusic import Tempo
        tempos.insert(0, Tempo(0, 120))

    def tick_to_seconds(tick):
        seconds, previous_tick, qpm = 0.0, 0, 120.0
        for tempo in tempos:
            if tempo.time >= tick: break
            seconds += (tempo.time - previous_tick) * 60.0 / (qpm * ticks_per_quarter)
            previous_tick, qpm = tempo.time, float(tempo.qpm)
        return seconds + (tick - previous_tick) * 60.0 / (qpm * ticks_per_quarter)

    notes = []
    for track in score.tracks:
        for note in track.notes:
            start = tick_to_seconds(note.time)
            notes.append((start, tick_to_seconds(note.time + note.duration),
                          int(note.pitch), int(note.velocity)))
    return notes

def synthesize_sine_preview(score: Any, path: Path, sample_rate=22050, max_duration=10.0):
    """Tạo WAV 5–10 giây; lặp motif ngắn thay vì đệm im lặng."""
    notes = score_notes_in_seconds(score)
    natural_duration = max((n[1] for n in notes), default=0.0)
    if natural_duration <= 0.0:
        raise ValueError("MIDI sinh ra không có nốt nhạc")
    duration = min(max(natural_duration, 5.0), max_duration)
    audio = np.zeros(max(1, int(duration * sample_rate)), dtype=np.float32)
    repeat_count = max(1, math.ceil(duration / natural_duration))
    for repeat_index in range(repeat_count):
        offset = repeat_index * natural_duration
        for start, end, pitch, velocity in notes:
            begin = int((start + offset) * sample_rate)
            finish = min(len(audio), int((end + offset) * sample_rate))
            count = finish - begin
            if count <= 0: continue
            t = np.arange(count, dtype=np.float32) / sample_rate
            env = np.ones(count, dtype=np.float32)
            fade = min(count // 2, int(.015 * sample_rate))
            if fade:
                env[:fade] = np.linspace(0, 1, fade); env[-fade:] = np.linspace(1, 0, fade)
            freq = 440 * 2 ** ((pitch - 69) / 12)
            audio[begin:finish] += .12 * velocity / 127 * env * np.sin(2 * math.pi * freq * t)
    # Fade-out 30 ms khi phải cắt ở giây thứ 10, tránh click nhưng không thêm im lặng.
    if natural_duration > max_duration:
        fade = min(len(audio), int(.03 * sample_rate))
        audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0
    if peak > .95: audio *= .95 / peak
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes((audio * 32767).astype("<i2").tobytes())
    return duration

def normalize_preview_duration(path: Path, min_duration=5.0, max_duration=10.0):
    """Lặp audio ngắn và cắt audio dài, giữ nguyên định dạng WAV của FluidSynth."""
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    frame_size = params.nchannels * params.sampwidth
    frame_count = len(frames) // frame_size
    if frame_count <= 0:
        raise ValueError("FluidSynth không tạo ra dữ liệu âm thanh")
    target_frames = min(max(frame_count, int(min_duration * params.framerate)),
                        int(max_duration * params.framerate))
    repeats = math.ceil(target_frames / frame_count)
    frames = (frames * repeats)[:target_frames * frame_size]
    with wave.open(str(path), "wb") as output:
        output.setparams(params)
        output.writeframes(frames)
    return target_frames / params.framerate

def synthesize_preview(score: Any, midi_path: Path, wav_path: Path):
    """Render MIDI bằng FluidSynth/SoundFont; chỉ fallback sóng sin khi thiếu binary."""
    if FLUIDSYNTH and FLUIDSYNTH.exists() and SOUNDFONT.exists():
        result = subprocess.run(
            [str(FLUIDSYNTH), "-ni", "-F", str(wav_path), "-r", "44100",
             "-g", "0.7", str(SOUNDFONT), str(midi_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"FluidSynth lỗi: {message}")
        return normalize_preview_duration(wav_path)
    return synthesize_sine_preview(score, wav_path)

@app.get("/")
def index(): return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "checkpoint":CHECKPOINT.exists(), "tokenizer":TOKENIZER_CONFIG.exists(),
            "model_loaded":service.model is not None, "device":str(service.device),
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda":torch.version.cuda,
            "audio_renderer":"fluidsynth" if FLUIDSYNTH and FLUIDSYNTH.exists() else "sine",
            "soundfont":SOUNDFONT.name if SOUNDFONT.exists() else None}

@app.post("/api/generate")
def generate(request: GenerateRequest):
    request.prompt = request.prompt.strip()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    item_id = uuid.uuid4().hex
    midi, wav = OUTPUT_DIR/f"{item_id}.mid", OUTPUT_DIR/f"{item_id}.wav"
    try:
        actual_steps = request.max_steps
        score = service.generate(request, midi)
        duration = synthesize_preview(score, midi, wav)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(507, "GPU hết VRAM; hãy giảm MUSICGEN_MAX_STEPS.") from exc
    except Exception as exc:
        midi.unlink(missing_ok=True); wav.unlink(missing_ok=True)
        raise HTTPException(500, f"Không thể tạo nhạc: {exc}") from exc
    sessions = read_history()
    session = next((x for x in sessions if x["id"] == request.session_id), None)
    if session is None:
        session = {"id":uuid.uuid4().hex, "title":" ".join(request.prompt.split()[:8]),
                   "created_at":now(), "messages":[]}
        sessions.append(session)
    metadata = {"mode":"model", "device":str(service.device), "max_steps":actual_steps,
                "duration_policy":"5_to_10_seconds_repeat_short_motif",
                "audio_renderer":"fluidsynth" if FLUIDSYNTH and FLUIDSYNTH.exists() else "sine"}
    audio_url, midi_url = f"/generated/{item_id}.wav", f"/generated/{item_id}.mid"
    session["messages"] += [{"role":"user", "content":request.prompt, "created_at":now()},
                            {"role":"assistant", "audio_url":audio_url, "midi_url":midi_url,
                             "duration":duration, "metadata":metadata, "created_at":now()}]
    session["updated_at"] = now(); write_history(sessions)
    return {"status":"ok", "session_id":session["id"], "audio_url":audio_url,
            "midi_url":midi_url, "duration":duration, "metadata":metadata}

@app.get("/api/history")
def list_history():
    result = [{"id":x["id"], "title":x["title"], "updated_at":x.get("updated_at",x.get("created_at")),
               "message_count":sum(m.get("role")=="assistant" for m in x.get("messages",[]))}
              for x in read_history()]
    return sorted(result, key=lambda x:x["updated_at"] or "", reverse=True)

@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    session = next((x for x in read_history() if x["id"] == session_id), None)
    if session is None: raise HTTPException(404, "Không tìm thấy hội thoại")
    return session

@app.delete("/api/history/{session_id}")
def delete_history(session_id: str):
    sessions = read_history(); remaining = [x for x in sessions if x["id"] != session_id]
    if len(remaining) == len(sessions): raise HTTPException(404, "Không tìm thấy hội thoại")
    write_history(remaining); return {"status":"ok"}

@app.delete("/api/history")
def clear_history(): write_history([]); return {"status":"ok"}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/generated", StaticFiles(directory=OUTPUT_DIR, check_dir=False), name="generated")

def main():
    uvicorn.run("server:app", host=os.getenv("MUSICGEN_HOST","127.0.0.1"),
                port=int(os.getenv("MUSICGEN_PORT","8000")), reload=False)

if __name__ == "__main__": main()
