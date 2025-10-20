import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------
# Configurações do sistema
# -----------------------

# Classes proibidas para o MVP (presentes no COCO): knife, scissors, bottle
PROHIBITED_CLASS_NAMES = {"knife", "scissors", "bottle"}

# limiar de confiança do detector
CONF_THRESHOLD = 0.45

# tempo mínimo (s) que o objeto deve permanecer na zona para disparar alerta
DWELL_SECONDS = 0.7

# tempo máximo (s) sem ver o mesmo objeto para "esquecer" o track simplificado
TRACK_TTL = 1.0

# Zonas proibidas em coordenadas normalizadas (0..1) no formato de polígonos
# Edite estes pontos conforme a sua câmera (use 3+ pontos por polígono)
ZONES_NORM = [
    {
        "name": "No-Blade Zone 1",
        "polygon": [(0.05, 0.60), (0.60, 0.60), (0.60, 0.95), (0.05, 0.95)]
    },
    # Exemplo de segunda zona:
    # {"name": "No-Blade Zone 2", "polygon": [(0.65, 0.10), (0.95, 0.10), (0.95, 0.50), (0.65, 0.50)]}
]

# arquivo de log CSV
LOG_FILE = "data/eventos_proibidos.csv"

# -----------------------
# Utilidades geométricas
# -----------------------

def scale_polygon(polynorm, w, h):
    return [(int(x * w), int(y * h)) for (x, y) in polynorm]

def point_in_polygon(x, y, polygon):
    # Algoritmo ray casting
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Checa interseção do segmento com o raio horizontal
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside

def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-9)

# -----------------------
# Track simplificado por IoU
# -----------------------

class TrackManager:
    def __init__(self, iou_thresh=0.35):
        self.tracks = []  # lista de dicts: {bbox, cls_name, first_seen, last_seen, dwell, alerted, zone}
        self.iou_thresh = iou_thresh

    def update(self, detections, now):
        """
        detections: lista de dicts {bbox, cls_name, zone_name} já filtradas por zona e classe
        """
        # Decai/remover tracks antigos
        new_tracks = []
        for tr in self.tracks:
            if now - tr["last_seen"] <= TRACK_TTL:
                new_tracks.append(tr)
        self.tracks = new_tracks

        # Associa por IoU
        used = [False] * len(detections)
        for tr in self.tracks:
            best = -1
            best_iou = 0.0
            for i, det in enumerate(detections):
                if used[i]:
                    continue
                if tr["cls_name"] != det["cls_name"] or tr["zone"] != det["zone"]:
                    continue
                iou_val = iou(tr["bbox"], det["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best = i
            if best != -1 and best_iou >= self.iou_thresh:
                # Match → atualiza track
                det = detections[best]
                used[best] = True
                # incrementa dwell pelo delta de tempo
                dt = now - tr["last_seen"]
                tr["dwell"] += max(0.0, dt)
                tr["bbox"] = det["bbox"]
                tr["last_seen"] = now

        # Cria novas tracks para não associados
        for i, det in enumerate(detections):
            if not used[i]:
                self.tracks.append({
                    "bbox": det["bbox"],
                    "cls_name": det["cls_name"],
                    "first_seen": now,
                    "last_seen": now,
                    "dwell": 0.0,
                    "alerted": False,
                    "zone": det["zone"]
                })

        return self.tracks

# -----------------------
# Log de eventos
# -----------------------

def ensure_log_header(path):
    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "zone", "class", "dwell_seconds", "bbox[x1,y1,x2,y2]"])

def log_event(path, zone, cls_name, dwell, bbox):
    ts = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, zone, cls_name, f"{dwell:.2f}", f"{[int(v) for v in bbox]}"])

# -----------------------
# Desenho
# -----------------------

def draw_polygon(frame, poly, name):
    cv2.polylines(frame, [np.array(poly, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    # label
    x = min(p[0] for p in poly)
    y = min(p[1] for p in poly) - 8
    cv2.putText(frame, name, (x, max(y, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

def draw_box(frame, bbox, label, color=(0, 0, 255)):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

# -----------------------
# Loop principal
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0 para webcam, caminho de vídeo, ou URL RTSP")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="pesos YOLO (COCO)")
    parser.add_argument("--imgsz", type=int, default=640, help="tamanho da imagem p/ inferência")
    parser.add_argument("--show", action="store_true", help="forçar janela visível (por padrão já mostra)")
    args = parser.parse_args()

    # Abre vídeo
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {args.source}")

    # Carrega modelo
    model = YOLO(args.weights)

    # Log
    ensure_log_header(LOG_FILE)

    # Track manager
    tracker = TrackManager(iou_thresh=0.35)

    last_shape = None
    zones_px = None

    fps_t0 = time.time()
    fps_frames = 0
    fps_val = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # (Re)escala zonas se tamanho mudar
        if last_shape != (w, h):
            zones_px = []
            for z in ZONES_NORM:
                zones_px.append({
                    "name": z["name"],
                    "polygon": scale_polygon(z["polygon"], w, h)
                })
            last_shape = (w, h)

        # Inferência
        results = model(frame, imgsz=args.imgsz, conf=CONF_THRESHOLD, verbose=False)
        res = results[0]
        names = res.names

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name not in PROHIBITED_CLASS_NAMES:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                # Verifica se centro está em alguma zona
                zone_name = None
                for z in zones_px:
                    if point_in_polygon(cx, cy, z["polygon"]):
                        zone_name = z["name"]
                        break
                if zone_name is None:
                    continue  # fora de zona proibida → ignora
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "cls_name": cls_name,
                    "zone": zone_name,
                    "conf": conf
                })

        now = time.time()
        tracks = tracker.update(detections, now)

        # Alerta e desenho
        for z in zones_px:
            draw_polygon(frame, z["polygon"], z["name"])

        for tr in tracks:
            label = f'{tr["cls_name"]} ({tr["zone"]})'
            color = (0, 0, 255) if not tr["alerted"] else (0, 165, 255)  # vermelho → alerta pendente, laranja → já logou
            draw_box(frame, tr["bbox"], label, color=color)

            if not tr["alerted"] and tr["dwell"] >= DWELL_SECONDS:
                tr["alerted"] = True
                log_event(LOG_FILE, tr["zone"], tr["cls_name"], tr["dwell"], tr["bbox"])
                # Beep simples no terminal
                print("\a", end="", flush=True)
                print(f"[ALERTA] {tr['cls_name']} na {tr['zone']} (dwell {tr['dwell']:.2f}s)")

        # FPS overlay
        fps_frames += 1
        if fps_frames >= 10:
            now2 = time.time()
            fps_val = fps_frames / (now2 - fps_t0 + 1e-9)
            fps_t0 = now2
            fps_frames = 0

        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 200, 40), 2, cv2.LINE_AA)
        cv2.imshow("Deteccao de Itens Proibidos - MVP", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
