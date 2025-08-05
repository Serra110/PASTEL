import os
import sys
import logging
import time
from datetime import datetime
import argparse
import urllib.request

# ===================== MODELO DNN =====================
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
PROTO = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

def download_model():
    if not os.path.exists(PROTO):
        print("A sacar deploy.prototxt...")
        urllib.request.urlretrieve(PROTO_URL, PROTO)
    if not os.path.exists(MODEL):
        print("A sacar res10_300x300_ssd_iter_140000.caffemodel...")
        urllib.request.urlretrieve(MODEL_URL, MODEL)

download_model()

# ===================== DEPENDÊNCIAS =====================
def install_and_import(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        pip_name = pip_name or package
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
        __import__(package)

for pkg, pipn in [('cv2', 'opencv-python'), ('numpy', 'numpy'), ('picamera2', 'picamera2')]:
    install_and_import(pkg, pipn)

import cv2
import numpy as np

# ===================== CONFIGURAÇÕES =====================
DEFAULT_CONFIDENCE = 0.5
DEFAULT_COOLDOWN = 10  # segundos
DEFAULT_RESOLUTION = (1920, 1080)
MIN_FACE_SIZE = 150  # px mínimo para considerar uma face real
FOTOS_DIR = '/home/diogo/projeto-PASTEL/fotosdetectadas'
LOGS_DIR = '/home/diogo/projeto-PASTEL/logs'
LOG_FILE = os.path.join(LOGS_DIR, 'app.log')

# ===================== ARGUMENTOS =====================
parser = argparse.ArgumentParser(description='Deteção de faces com autofocus físico (Picamera2) + OpenCV DNN')
parser.add_argument('--debug', action='store_true', help='Mostrar preview com deteção')
parser.add_argument('--video', action='store_true', help='Modo vídeo interativo (usa ficheiro de vídeo em vez da câmara)')
parser.add_argument('--cooldown', type=int, default=DEFAULT_COOLDOWN, help='Cooldown entre fotos da mesma face (s)')
parser.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE, help='Confiança mínima para deteção de face')
parser.add_argument('--resolution', type=str, default='1920x1080', help='Resolução da câmara (ex: 1920x1080)')
args = parser.parse_args()

# ===================== LOGGING =====================
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FOTOS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def unique_filename(base_dir, prefix='face', ext='jpg'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return os.path.join(base_dir, f"{prefix}_{ts}.{ext}")

def log_action(action, info=None):
    msg = f"{action}"
    if info:
        msg += f" | {info}"
    logging.info(msg)

def abort_if_low_disk(path, min_free_mb=100):
    statvfs = os.statvfs(path)
    free_mb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 * 1024)
    if free_mb < min_free_mb:
        logging.error(f'Espaço em disco insuficiente! ({free_mb:.1f} MB livres)')
        sys.exit(1)

def faces_are_close(face1, face2, threshold=0.2):
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    center1 = (x1 + w1/2, y1 + h1/2)
    center2 = (x2 + w2/2, y2 + h2/2)
    dist = np.linalg.norm(np.array(center1) - np.array(center2))
    max_dim = max(w1, h1, w2, h2)
    return dist < threshold * max_dim

# ===================== DNN FACE DETECTOR =====================
net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

def detect_faces_dnn(frame, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# ===================== CICLO PRINCIPAL =====================
def process_stream_from_camera(width, height, cooldown, confidence, debug, fotos_dir):
    from picamera2 import Picamera2
    last_faces = []   # [(bbox, timestamp)]
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (width, height)})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # tempo para autofocus inicial

    # Ativar autofocus contínuo (Camera Module 3)
    try:
        picam2.set_controls({"AfMode": 2})  # 2 = Continuous autofocus
        log_action("Autofocus físico ativado (contínuo)")
    except Exception as e:
        log_action("Não foi possível ativar autofocus físico", str(e))

    while True:
        # Forçar autofocus antes de cada frame (opcional, pode ser removido se já estiver em modo contínuo)
        try:
            picam2.set_controls({"AfTrigger": 0})  # Trigger autofocus
        except Exception:
            pass

        frame = picam2.capture_array()
        now = time.time()
        new_faces = []
        faces = detect_faces_dnn(frame, confidence)
        for (x, y, w, h) in faces:
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue  # ignora detecções pequenas (olhos, etc)
            recently_seen = False
            for (old, ts) in last_faces:
                if faces_are_close((x, y, w, h), old) and now - ts < cooldown:
                    recently_seen = True
                    break
            if recently_seen:
                continue
            crop = frame[y:y+h, x:x+w]
            # Converter de BGR para RGB antes de guardar
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            fname = unique_filename(fotos_dir, prefix='face')
            cv2.imwrite(fname, crop_rgb)
            log_action('Face detetada', fname)
            new_faces.append(((x, y, w, h), now))
        last_faces = [item for item in last_faces if now - item[1] < cooldown] + new_faces
        # Preview
        if debug and os.environ.get('DISPLAY'):
            preview = frame.copy()
            for (x, y, w, h) in faces:
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                cv2.rectangle(preview, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.imshow('Preview', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    try:
        picam2.close()
    except Exception:
        pass
    if os.environ.get('DISPLAY'):
        cv2.destroyAllWindows()

def process_stream_from_video(width, height, cooldown, confidence, debug, fotos_dir):
    last_faces = []   # [(bbox, timestamp)]
    while True:
        video_path = input('Caminho do vídeo a analisar (ou ENTER para sair): ').strip()
        if not video_path:
            print('A terminar. Até logo!')
            break
        if not os.path.isfile(video_path):
            print('Ficheiro não encontrado. Tenta de novo.')
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('Não foi possível abrir o vídeo.')
            continue
        log_action('Análise de vídeo iniciada', video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            now = time.time()
            new_faces = []
            faces = detect_faces_dnn(frame, confidence)
            for (x, y, w, h) in faces:
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                recently_seen = False
                for (old, ts) in last_faces:
                    if faces_are_close((x, y, w, h), old) and now - ts < cooldown:
                        recently_seen = True
                        break
                if recently_seen:
                    continue
                crop = frame[y:y+h, x:x+w]
                # Converter de BGR para RGB antes de guardar
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                fname = unique_filename(fotos_dir, prefix='face')
                cv2.imwrite(fname, crop_rgb)
                log_action('Face detetada', fname)
                new_faces.append(((x, y, w, h), now))
            last_faces = [item for item in last_faces if now - item[1] < cooldown] + new_faces
            # Preview
            if debug and os.environ.get('DISPLAY'):
                preview = frame.copy()
                for (x, y, w, h) in faces:
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        continue
                    cv2.rectangle(preview, (x, y), (x+w, y+h), (255,0,0), 2)
                cv2.imshow('Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        log_action('Análise de vídeo terminada', video_path)
        print('Vídeo analisado. Queres analisar outro?')
    if os.environ.get('DISPLAY'):
        cv2.destroyAllWindows()

def main():
    width, height = map(int, args.resolution.split('x'))
    abort_if_low_disk(FOTOS_DIR)
    if args.video:
        process_stream_from_video(width, height, args.cooldown, args.confidence, args.debug, FOTOS_DIR)
    else:
        process_stream_from_camera(width, height, args.cooldown, args.confidence, args.debug, FOTOS_DIR)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f'Erro inesperado: {e}')
        sys.exit(1)