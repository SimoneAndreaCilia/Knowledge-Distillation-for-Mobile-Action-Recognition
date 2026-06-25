"""
Script per testare il modello su una singola clip video.

Esempio di utilizzo:
python -m src.evaluation.inference --video "data/hmdb51/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi" --model student --checkpoint "checkpoints/distilled_AT_T10_seed1234/best_model.pth"
"""

import argparse
import os
import torch
import cv2
import numpy as np

# Aggiungiamo src al path per poter importare i moduli
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models import build_model

# Statistiche Kinetics-400 per la normalizzazione
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

def preprocess_video(video_path, num_frames=16, target_size=112, return_frames=False):
    """
    Carica il video e applica lo stesso preprocessing usato in fase di test:
    - Campionamento deterministico di num_frames
    - Center crop spaziale
    - Normalizzazione con medie/std di Kinetics
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV usa BGR, convertiamo in RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Impossibile leggere il video o video vuoto: {video_path}")
        
    frames = np.stack(frames)
    total_frames = len(frames)
    
    # Campionamento temporale
    if total_frames >= num_frames:
        segment_length = total_frames / num_frames
        # Prende il frame al centro di ogni segmento temporale
        indices = [int((start + end) // 2) for start, end in zip(
            np.arange(0, total_frames, segment_length),
            np.arange(segment_length, total_frames + segment_length, segment_length)
        )][:num_frames]
        indices = np.clip(indices, 0, total_frames - 1)
        frames = frames[indices]
    else:
        # Se il video è più corto, ripetiamo i frame interpolando gli indici
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        indices = np.clip(indices, 0, total_frames - 1)
        frames = frames[indices]

    # Resize e Center Crop
    resize_size = int(target_size * 1.14)
    resized = np.stack([cv2.resize(f, (resize_size, resize_size)) for f in frames])
    
    new_h, new_w = resized.shape[1], resized.shape[2]
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    cropped_frames = resized[:, top:top + target_size, left:left + target_size, :]
    
    clip = cropped_frames.astype(np.float32) / 255.0
    
    # (T, H, W, C) -> (C, T, H, W) e normalizzazione
    clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()
    mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)
    clip_tensor = (clip_tensor - mean) / std
    
    if return_frames:
        return clip_tensor.unsqueeze(0), cropped_frames
    return clip_tensor.unsqueeze(0) # Aggiungiamo la dimensione di batch

def main():
    parser = argparse.ArgumentParser(description="Script per testare il modello su una singola clip video")
    parser.add_argument("--video", type=str, required=True, help="Percorso del file video (es. .avi o .mp4)")
    parser.add_argument("--model", type=str, default="student", choices=["student", "teacher"], help="Tipo di modello")
    parser.add_argument("--checkpoint", type=str, required=True, help="Percorso del checkpoint .pth")
    parser.add_argument("--num_classes", type=int, default=51, help="Numero di classi")
    parser.add_argument("--width_mult", type=float, default=1.0, help="Moltiplicatore di larghezza per lo Student")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")
    
    # 1. Caricamento Modello
    print("Caricamento del modello...")
    model = build_model(
        model_name=args.model, 
        num_classes=args.num_classes, 
        pretrained=False,
        width_mult=args.width_mult
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint) # In caso sia stato salvato solo il dict
        
    model.to(device)
    model.eval()
    
    # 2. Preprocessing Video
    print("Elaborazione del video...")
    clip_tensor = preprocess_video(args.video).to(device)
    
    # 3. Inferenza
    print("Esecuzione dell'inferenza...\n")
    with torch.no_grad():
        logits = model(clip_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top5_probs, top5_classes = torch.topk(probs, 5, dim=1)
        
    print("=== Top 5 Predizioni ===")
    
    # Provo a caricare i nomi delle classi leggendo le cartelle in data/hmdb51
    class_names = []
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hmdb51")
    if os.path.exists(data_dir):
        folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        class_names = sorted(folders)
        
    for i in range(5):
        c_idx = top5_classes[0][i].item()
        c_prob = top5_probs[0][i].item() * 100
        
        if len(class_names) == args.num_classes:
            c_name = class_names[c_idx]
            print(f"{i+1}. Classe: {c_name:20s} | Confidenza: {c_prob:5.2f}%")
        else:
            print(f"{i+1}. Indice Classe: {c_idx:2d} | Confidenza: {c_prob:5.2f}%")
        
    if len(class_names) != args.num_classes:
        print("\nNota: gli indici classe (0-50) corrispondono all'ordine alfabetico delle cartelle del dataset HMDB51.")


if __name__ == "__main__":
    main()
