"""MusicGen2.0 API: text prompt -> MIDI -> lightweight WAV preview."""
from __future__ import annotations

import hashlib, json, math, os, re, shutil, subprocess, threading, time, uuid, wave
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
    max_steps: int = Field(default=int(os.getenv("MUSICGEN_MAX_STEPS", "384")), ge=32, le=768)
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
            candidate_count = max(1, min(4, int(os.getenv("MUSICGEN_CANDIDATES", "3"))))
            profiles = [(0.82, 32, 2.2), (1.0, 64, 3.0), (1.16, 96, 3.8), (0.92, 48, 1.6)][:candidate_count]
            prompts = [request.prompt] * candidate_count
            temperatures = [max(.1, min(2.0, request.temperature * p[0])) for p in profiles]
            top_ks = [max(0, min(200, p[1] if request.top_k == 50 else request.top_k)) for p in profiles]
            cfg_scales = [max(0.0, min(8.0, p[2] if request.cfg_scale == 3.0 else request.cfg_scale)) for p in profiles]
            tokens = model.generate(prompts, max_steps=request.max_steps, temperature=temperatures,
                                    top_k=top_ks, cfg_scale=cfg_scales)
            candidate_paths = [midi_path.with_name(f"{midi_path.stem}.candidate{i}.mid")
                               for i in range(candidate_count)]
            model.save_tokens_to_midi(tokens, str(TOKENIZER_CONFIG), [str(path) for path in candidate_paths])
        from symusic import Score
        ranked = []
        try:
            for index, path in enumerate(candidate_paths):
                candidate_score = Score(str(path))
                ranked.append((midi_candidate_score(candidate_score, request.prompt), index, path, candidate_score))
            ranked.sort(key=lambda item: item[0], reverse=True)
            best_score, best_index, best_path, score = ranked[0]
            shutil.copyfile(best_path, midi_path)
        finally:
            for path in candidate_paths:
                path.unlink(missing_ok=True)
        constraints = apply_prompt_constraints(score, request.prompt)
        score.dump_midi(str(midi_path))
        quality = ensure_musical_score(score, midi_path, request.prompt)
        quality["prompt_constraints"] = constraints
        quality["checkpoint_search"] = {
            "candidates": candidate_count, "selected": best_index,
            "scores": [round(item[0], 2) for item in sorted(ranked, key=lambda item: item[1])],
            "temperatures": temperatures, "top_k": top_ks, "cfg_scale": cfg_scales,
            "max_steps": request.max_steps,
        }
        return score, quality

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

def midi_quality(score: Any):
    """Detect collapsed REMI output: stacked notes, tiny duration or almost one pitch."""
    notes = [note for track in score.tracks for note in track.notes]
    tpq = max(1, int(score.ticks_per_quarter))
    if not notes:
        return {"valid": False, "notes": 0, "pitches": 0, "onsets": 0, "span_beats": 0.0}
    onsets = {int(note.time) for note in notes}
    pitches = {int(note.pitch) for note in notes}
    start = min(int(note.time) for note in notes)
    end = max(int(note.time + note.duration) for note in notes)
    span_beats = max(0.0, (end - start) / tpq)
    # At least two bars, a real rhythmic contour and enough tonal variety.
    valid = len(notes) >= 8 and len(onsets) >= 8 and len(pitches) >= 4 and span_beats >= 8.0
    return {"valid": valid, "notes": len(notes), "pitches": len(pitches),
            "onsets": len(onsets), "span_beats": round(span_beats, 2)}

def midi_candidate_score(score: Any, prompt: str):
    """Rank checkpoint samples by musical structure and explicit prompt constraints."""
    quality = midi_quality(score)
    notes = [note for track in score.tracks for note in track.notes]
    if not notes:
        return -1000.0
    controls = parse_prompt_controls(prompt)
    programs = {int(track.program) for track in score.tracks if not track.is_drum}
    has_drums = any(track.is_drum for track in score.tracks)
    onset_counts = {}
    for note in notes:
        onset_counts[int(note.time)] = onset_counts.get(int(note.time), 0) + 1
    max_stack = max(onset_counts.values(), default=0)
    score_value = 0.0
    score_value += min(40, quality["onsets"]) * 1.8
    score_value += min(16, quality["pitches"]) * 2.5
    score_value += max(0.0, 28.0 - abs(16.0 - quality["span_beats"]) * 2.0)
    score_value += 80.0 if quality["valid"] else 0.0
    score_value -= max(0, max_stack - 6) * 4.0
    for _, program in controls["requested"]:
        score_value += 35.0 if program in programs else -12.0
    if controls["bass"]:
        score_value += 30.0 if controls["bass"][1] in programs else -10.0
    if controls["no_drums"] or controls["solo"]:
        score_value += 18.0 if not has_drums else -45.0
    elif "drum" in controls["text"] or "percussion" in controls["text"]:
        score_value += 25.0 if has_drums else -18.0
    return score_value

def parse_prompt_controls(prompt: str):
    """Extract concrete musical constraints instead of treating the prompt as a genre label."""
    text = prompt.lower().replace("♭", "b").replace("♯", "#")
    programs = {
        "grand piano": 0, "acoustic piano": 0, "piano": 0, "rhodes": 4,
        "electric piano": 4, "organ": 19, "marimba": 12, "vibraphone": 11,
        "acoustic guitar": 24, "nylon guitar": 24, "electric guitar": 27,
        "distorted guitar": 30, "guitar": 25, "violin": 40, "viola": 41,
        "cello": 42, "strings": 48, "trumpet": 56, "trombone": 57,
        "french horn": 60, "brass": 61, "saxophone": 65, "sax": 65,
        "flute": 73, "clarinet": 71, "synth lead": 80, "synth": 80,
        "pad": 89, "choir": 52, "bells": 10,
    }
    bass_programs = {"synth bass": 38, "electric bass": 33, "acoustic bass": 32,
                     "contrabass": 43, "bass guitar": 33, "bass": 33}
    excluded = set()
    for name in (*programs, *bass_programs, "drums", "drum", "percussion", "melody"):
        if re.search(rf"\b(?:no|without)\s+(?:[a-z-]+\s+)?{re.escape(name)}\b", text):
            excluded.add(name)
    requested = [(name, program) for name, program in programs.items()
                 if name in text and name not in excluded]
    # Preserve prompt order; for overlapping names prefer the specific phrase.
    requested.sort(key=lambda item: (text.find(item[0]), -len(item[0])))
    deduped, seen_programs = [], set()
    for item in requested:
        if item[1] not in seen_programs:
            deduped.append(item); seen_programs.add(item[1])
    requested_bass = next(((name, program) for name, program in bass_programs.items()
                           if name in text and name not in excluded), None)
    bpm_match = re.search(r"\b(\d{2,3})\s*(?:bpm|beats? per minute)\b", text)
    bpm = max(45, min(200, int(bpm_match.group(1)))) if bpm_match else None
    key_names = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4,
                 "f": 5, "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8,
                 "a": 9, "a#": 10, "bb": 10, "b": 11}
    key_match = re.search(r"\b(?:in\s+)?([a-g](?:#|b)?)\s+(major|minor)\b", text)
    key = key_names.get(key_match.group(1)) if key_match else None
    mode = key_match.group(2) if key_match else ("minor" if any(w in text for w in
           ("minor", "sad", "dark", "melanch", "dramatic", "myster")) else "major")
    energy = 1.18 if any(w in text for w in ("aggressive", "powerful", "intense", "energetic", "hard-hitting")) \
        else 0.82 if any(w in text for w in ("soft", "gentle", "calm", "mellow", "relax", "ambient")) else 1.0
    return {"text": text, "requested": deduped, "bass": requested_bass, "excluded": excluded,
            "bpm": bpm, "key": key, "mode": mode, "energy": energy,
            "solo": "solo" in text, "no_melody": "melody" in excluded,
            "no_bass": any(name in excluded for name in bass_programs),
            "no_drums": any(name in excluded for name in ("drum", "drums", "percussion"))}

def apply_prompt_constraints(score: Any, prompt: str):
    """Preserve checkpoint composition while enforcing explicit instruments, tempo and key."""
    from symusic import Tempo
    controls = parse_prompt_controls(prompt)
    tracks = list(score.tracks)
    melodic = [track for track in tracks if not track.is_drum]
    removed = []

    if controls["no_drums"] or controls["solo"]:
        removed.extend(track.name for track in tracks if track.is_drum)
        tracks = [track for track in tracks if not track.is_drum]
    if controls["no_bass"]:
        bass_programs = set(range(32, 40)) | {43}
        removed.extend(track.name for track in tracks if not track.is_drum and int(track.program) in bass_programs)
        tracks = [track for track in tracks if track.is_drum or int(track.program) not in bass_programs]

    melodic = [track for track in tracks if not track.is_drum]
    assigned = []
    if controls["requested"] and melodic:
        for index, (name, program) in enumerate(controls["requested"]):
            if index >= len(melodic):
                break
            melodic[index].program = program
            melodic[index].name = name.title()
            assigned.append({"track": name.title(), "program": program})
    if controls["bass"]:
        candidates = [track for track in melodic if track not in melodic[:len(controls["requested"])]]
        if candidates:
            candidates[-1].program = controls["bass"][1]
            candidates[-1].name = controls["bass"][0].title()
            assigned.append({"track": candidates[-1].name, "program": controls["bass"][1]})

    if controls["solo"] and melodic:
        keep = melodic[0]
        removed.extend(track.name for track in tracks if track is not keep)
        tracks = [keep]

    score.tracks.clear()
    score.tracks.extend(tracks)
    if controls["bpm"]:
        score.tempos.clear()
        score.tempos.append(Tempo(0, controls["bpm"]))

    snapped = 0
    if controls["key"] is not None:
        scale = (0, 2, 3, 5, 7, 8, 10) if controls["mode"] == "minor" else (0, 2, 4, 5, 7, 9, 11)
        allowed = {(controls["key"] + interval) % 12 for interval in scale}
        tonal_notes = [note for track in score.tracks if not track.is_drum for note in track.notes]
        for index, note in enumerate(sorted(tonal_notes, key=lambda n: (int(n.time), int(n.pitch)))):
            pitch = int(note.pitch)
            nearest = min(range(-2, 3), key=lambda delta: (0 if (pitch + delta) % 12 in allowed else 10) + abs(delta))
            new_pitch = max(21, min(108, pitch + nearest))
            if new_pitch != pitch:
                note.pitch = new_pitch; snapped += 1
        # Token collapse can leave a valid rhythm with only one or two notes. Keep the
        # checkpoint rhythm but unfold its pitches inside the requested key.
        if tonal_notes and len({int(note.pitch) for note in tonal_notes}) < 4:
            ordered = sorted(tonal_notes, key=lambda n: (int(n.time), int(n.pitch)))
            root = 60 + controls["key"]
            contour = (0, 2, 4, 2, 1, 3, 5, 3)
            for index, note in enumerate(ordered):
                octave = 12 * max(-2, min(2, (int(note.pitch) - root) // 12))
                note.pitch = max(21, min(108, root + octave + scale[contour[index % len(contour)] % 7]))

    # Fit the complete checkpoint phrase into the audible 5–10 second preview while
    # preserving all relative timing and durations.
    all_notes = [note for track in score.tracks for note in track.notes]
    time_scale = 1.0
    if all_notes:
        tpq = max(1, int(score.ticks_per_quarter))
        first_tick = min(int(note.time) for note in all_notes)
        last_tick = max(int(note.time + note.duration) for note in all_notes)
        span = max(1, last_tick - first_tick)
        effective_bpm = controls["bpm"] or (float(score.tempos[0].qpm) if score.tempos else 120.0)
        target_ticks = int(tpq * max(8.0, min(24.0, effective_bpm * 8.0 / 60.0)))
        if span > target_ticks:
            time_scale = target_ticks / span
        for note in all_notes:
            note.time = max(0, int((int(note.time) - first_tick) * time_scale))
            note.duration = max(1, int(int(note.duration) * time_scale))

    return {"bpm": controls["bpm"], "key": controls["key"], "mode": controls["mode"],
            "assigned_instruments": assigned, "removed_tracks": removed, "snapped_notes": snapped,
            "time_scale": round(time_scale, 4)}

def repair_collapsed_score(score: Any, prompt: str):
    """Build a prompt-aware arrangement when the neural MIDI stream collapses."""
    from symusic import Note, Tempo, Track

    tpq = max(1, int(score.ticks_per_quarter))
    controls = parse_prompt_controls(prompt)
    text = controls["text"]
    seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "big")
    source = sorted((note for track in score.tracks for note in track.notes),
                    key=lambda note: (int(note.time), int(note.pitch)))
    source_pitches = [int(note.pitch) for note in source if 36 <= int(note.pitch) <= 96]
    center = sorted(source_pitches)[len(source_pitches) // 2] if source_pitches else 60 + seed % 12
    root_class = controls["key"] if controls["key"] is not None else seed % 12
    root = 48 + root_class

    drum_only = any(word in text for word in ("drum only", "drum beat", "solo drum", "percussion only", "trống")) \
        or ("drum" in text and any(word in text for word in ("no melody", "without melody", "no bass")))
    if any(word in text for word in ("rock", "metal", "guitar", "punk")):
        style, lead_program, harmony_program, bass_program = "rock", 30, 29, 33
    elif any(word in text for word in ("synth", "electronic", "cyberpunk", "neon", "techno", "edm")):
        style, lead_program, harmony_program, bass_program = "synthwave", 80, 89, 38
    elif any(word in text for word in ("tropical", "marimba", "island", "summer", "reggae")):
        style, lead_program, harmony_program, bass_program = "tropical", 12, 24, 33
    elif any(word in text for word in ("cinematic", "orchestra", "epic", "heroic", "film score")):
        style, lead_program, harmony_program, bass_program = "cinematic", 48, 60, 43
    elif any(word in text for word in ("lo-fi", "lofi", "chill", "jazz", "rhodes")):
        style, lead_program, harmony_program, bass_program = "lofi", 4, 26, 33
    else:
        style, lead_program, harmony_program, bass_program = "acoustic", 24, 40, 32

    default_bpms = {"rock": 142, "synthwave": 118, "tropical": 124,
                    "cinematic": 96, "lofi": 82, "acoustic": 108}
    bpm = controls["bpm"] or default_bpms[style]
    score.tempos.clear()
    score.tempos.append(Tempo(0, bpm))
    if controls["requested"]:
        lead_program = controls["requested"][0][1]
        if len(controls["requested"]) > 1:
            harmony_program = controls["requested"][1][1]
    if controls["bass"]:
        bass_program = controls["bass"][1]

    # A real GM drum track: kick, snare, hats and a prompt-dependent fill.
    drum_notes = []
    kick_patterns = {
        "rock": {0, 4, 8, 10, 12}, "synthwave": {0, 4, 8, 12},
        "tropical": {0, 6, 10, 14}, "cinematic": {0, 8, 12},
        "lofi": {0, 7, 10}, "acoustic": {0, 5, 8, 13},
    }
    pattern = kick_patterns.get(style, kick_patterns["acoustic"])
    for bar in range(4):
        base = bar * 4 * tpq
        for slot in range(16):
            when = base + slot * max(1, tpq // 4)
            drum_notes.append(Note(when, max(1, tpq // 8), 42 if slot % 2 == 0 else 44, 54 + (slot % 4) * 3))
            if slot in pattern:
                drum_notes.append(Note(when, max(1, tpq // 5), 36, 104))
            if slot in (4, 12):
                drum_notes.append(Note(when, max(1, tpq // 5), 38, 108))
        if bar == 3:
            for offset, pitch in enumerate((45, 47, 48, 50)):
                drum_notes.append(Note(base + (12 + offset) * max(1, tpq // 4), max(1, tpq // 5), pitch, 96))

    score.tracks.clear()
    if drum_only:
        score.tracks.append(Track(name="Acoustic Drum Kit", program=0, is_drum=True, notes=drum_notes))
        return

    scale = (0, 2, 3, 5, 7, 8, 10) if controls["mode"] == "minor" else (0, 2, 4, 5, 7, 9, 11)
    step = max(1, tpq // (1 if style == "cinematic" else 2))
    note_count = 16 if style == "cinematic" else 32
    contours = {
        "rock": (0, 0, 2, 0, 3, 2, 4, 2), "synthwave": (0, 2, 4, 6, 4, 2, 5, 3),
        "tropical": (0, 2, 3, 5, 4, 2, 1, 3), "cinematic": (0, 1, 3, 4, 5, 3, 6, 5),
        "lofi": (0, 2, 1, 4, 3, 1, 2, 0), "acoustic": (0, 1, 2, 4, 3, 2, 1, 3),
    }
    contour = contours[style]
    melody_notes = []
    for index in range(note_count):
        source_pitch = source_pitches[index % len(source_pitches)] if source_pitches else center
        nearest = min(range(7), key=lambda i: abs((root + scale[i]) - source_pitch))
        degree = (nearest + contour[(index + seed) % len(contour)] + index // 16) % 7
        pitch = max(48, min(88, root + scale[degree] + (12 if index % 16 in (7, 15) else 0)))
        when = index * step
        if style in ("tropical", "lofi") and index % 4 in (1, 3):
            when += max(1, step // 3)
        duration = max(1, int(step * (0.55 if style in ("rock", "tropical") else 0.86)))
        velocity = max(35, min(120, int((82 + (18 if index % 8 == 0 else 0)) * controls["energy"])))
        melody_notes.append(Note(when, duration, pitch, velocity))

    chord_notes, bass_notes = [], []
    progressions = {
        "rock": (0, 5, 3, 4), "synthwave": (0, 3, 5, 4), "tropical": (0, 4, 5, 3),
        "cinematic": (0, 5, 2, 4), "lofi": (0, 3, 1, 4), "acoustic": (0, 4, 5, 3),
    }
    for bar, degree in enumerate(progressions[style]):
        bar_time = bar * 4 * tpq
        tones = [root + scale[degree], root + scale[(degree + 2) % 7] + (12 if degree + 2 >= 7 else 0),
                 root + scale[(degree + 4) % 7] + (12 if degree + 4 >= 7 else 0)]
        if style == "rock":
            tones = (tones[0], tones[2])  # power chords
        for pitch in tones:
            chord_velocity = max(30, min(105, int(62 * controls["energy"])))
            chord_notes.append(Note(bar_time, max(1, int((1.8 if style == "rock" else 3.8) * tpq)), max(40, min(80, pitch)), chord_velocity))
            if style == "rock":
                chord_notes.append(Note(bar_time + 2 * tpq, max(1, int(1.8 * tpq)), max(40, min(80, pitch)), max(35, min(112, int(68 * controls["energy"])))))
        bass_pitch = max(28, tones[0] - 12)
        bass_beats = (0, 1, 2, 3) if style in ("rock", "synthwave") else (0, 2)
        for beat in bass_beats:
            bass_notes.append(Note(bar_time + beat * tpq, max(1, int(.8 * tpq)), bass_pitch,
                                   max(35, min(115, int(78 * controls["energy"])))))

    requested_name = controls["requested"][0][0].title() if controls["requested"] else style.title()
    if not controls["no_melody"]:
        score.tracks.append(Track(name=f"{requested_name} Lead", program=lead_program, notes=melody_notes))
    # A true solo request must not silently add an orchestra behind it.
    if not controls["solo"] and not controls["no_melody"]:
        harmony_name = controls["requested"][1][0].title() if len(controls["requested"]) > 1 else style.title()
        score.tracks.append(Track(name=f"{harmony_name} Harmony", program=harmony_program, notes=chord_notes))
    if not controls["solo"] and not controls["no_bass"]:
        bass_name = controls["bass"][0].title() if controls["bass"] else style.title()
        score.tracks.append(Track(name=f"{bass_name} Bass", program=bass_program, notes=bass_notes))
    if not controls["solo"] and not controls["no_drums"]:
        score.tracks.append(Track(name="Drum Kit", program=0, is_drum=True, notes=drum_notes))

def ensure_musical_score(score: Any, midi_path: Path, prompt: str):
    before = midi_quality(score)
    repaired = not before["valid"]
    if repaired:
        repair_collapsed_score(score, prompt)
        score.dump_midi(str(midi_path))
    after = midi_quality(score)
    if not after["valid"]:
        raise ValueError(f"MIDI không đạt chất lượng tối thiểu: {after}")
    return {"repaired": repaired, "before": before, "after": after}

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
    """Bỏ khoảng lặng đầu, tăng âm lượng và đưa preview về khoảng 5–10 giây."""
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    if params.sampwidth != 2:
        raise ValueError(f"WAV FluidSynth có sample width không hỗ trợ: {params.sampwidth}")
    audio = np.frombuffer(frames, dtype="<i2").reshape(-1, params.nchannels)
    if not audio.size:
        raise ValueError("FluidSynth không tạo ra dữ liệu âm thanh")

    # Model đôi khi sinh nốt đầu tiên rất muộn. Cắt phần im lặng trước nốt thay vì
    # lấy 10 giây đầu của file và vô tình trả về một WAV hoàn toàn không có tiếng.
    level = np.max(np.abs(audio.astype(np.int32)), axis=1)
    peak = int(level.max())
    if peak <= 0:
        raise ValueError("FluidSynth tạo ra WAV không có tín hiệu")
    active = np.flatnonzero(level >= max(1, peak // 1000))
    lead_in = int(0.05 * params.framerate)
    start = max(0, int(active[0]) - lead_in)
    end = min(len(audio), int(active[-1]) + lead_in + 1)
    audio = audio[start:end].astype(np.float32)

    # MIDI có velocity rất nhỏ vẫn phải nghe được trên trình duyệt.
    peak = float(np.max(np.abs(audio)))
    audio *= (0.9 * np.iinfo(np.int16).max) / peak
    audio = np.clip(audio, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype("<i2")

    frame_count = len(audio)
    target_frames = min(max(frame_count, int(min_duration * params.framerate)),
                        int(max_duration * params.framerate))
    repeats = math.ceil(target_frames / frame_count)
    audio = np.tile(audio, (repeats, 1))[:target_frames]
    with wave.open(str(path), "wb") as output:
        output.setparams(params)
        output.writeframes(audio.tobytes())
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
        score, quality = service.generate(request, midi)
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
                "duration_policy":"5_to_10_seconds_quality_gated_arrangement",
                "midi_quality":quality,
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
