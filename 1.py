import cv2
import torch
import numpy as np
import serial  # Importar la biblioteca para comunicación serial
from ultralytics import YOLO
import time
import os
import json
from threading import Thread
import tkinter as tk
from tkinter import ttk

# Configuración avanzada
config = {
    'model_path': 'best.pt',
    'confidence_threshold': 0.3,
    'brightness': 70,
    'contrast': 60,
    'exposure': 50,
    'focus': 30,
    'resolution': (640, 480),
    'scale_factor': 1.0,
    'nms_iou_threshold': 0.4,
    'enable_edge_filter': False,
    'test_images_dir': 'test_images',
    'log_file': 'performance_log.json',
    'no_detection_timeout': 20,
    'auto_adjust_lighting': True,
    'serial_port': 'COM3',  # Cambia esto al puerto serial correcto
    'baud_rate': 9600        # Configuración de la velocidad del puerto
}

# Inicializar comunicación serial
serial_connection = serial.Serial(config['serial_port'], config['baud_rate'], timeout=1)

# Función para guardar configuración
def save_config(config_file='config.json'):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

# Función para cargar configuración
def load_config(config_file='config.json'):
    global config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

# Inicializar y cargar el modelo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(config['model_path']).to(device)

# Función para ajustar brillo y contraste
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    brightness = int((brightness - 50) * 2.55)
    contrast = int((contrast - 50) * 2.55)
    img = np.int16(image)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

# Función para aplicar filtro de borde
def apply_edge_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

# Función para ajustar automáticamente los parámetros
def adjust_parameters(frame):
    global config
    if config['auto_adjust_lighting']:
        avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if avg_brightness < 100:
            config['brightness'] = min(config['brightness'] + 5, 100)
        else:
            config['brightness'] = max(config['brightness'] - 5, 0)

    config['contrast'] = min(max(config['contrast'] + 5, 0), 100)
    save_config()

# Función para realizar pruebas con imágenes
def test_model_with_params(frame, config):
    frame = adjust_brightness_contrast(frame, brightness=config['brightness'], contrast=config['contrast'])
    if config['enable_edge_filter']:
        frame = apply_edge_filter(frame)

    frame_resized = cv2.resize(frame, (int(config['resolution'][0] * config['scale_factor']),
                                       int(config['resolution'][1] * config['scale_factor'])))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        results = model(frame_tensor)

    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())
        confidences.extend(result.boxes.conf.cpu().numpy())
        class_ids.extend(result.boxes.cls.cpu().numpy())

    keep = cv2.dnn.NMSBoxes(boxes, confidences, config['confidence_threshold'], config['nms_iou_threshold'])
    return keep.flatten() if len(keep) > 0 else []

# Función para evaluar rendimiento
def evaluate_model():
    global config
    best_fps = 0
    best_params = config.copy()

    for conf_thresh in [0.2, 0.3, 0.4, 0.5]:
        for bright in [50, 60, 70, 80]:
            for cont in [50, 60, 70, 80]:
                for scale in [0.5, 1.0, 1.5]:
                    for nms_thresh in [0.3, 0.4, 0.5]:
                        for edge_filter in [False, True]:
                            temp_config = config.copy()
                            temp_config.update({
                                'confidence_threshold': conf_thresh,
                                'brightness': bright,
                                'contrast': cont,
                                'scale_factor': scale,
                                'nms_iou_threshold': nms_thresh,
                                'enable_edge_filter': edge_filter
                            })

                            test_image_files = [f for f in os.listdir(temp_config['test_images_dir']) if f.endswith('.jpg') or f.endswith('.png')]
                            fps_values = []
                            for test_image_file in test_image_files:
                                img_path = os.path.join(temp_config['test_images_dir'], test_image_file)
                                frame = cv2.imread(img_path)
                                if frame is not None:
                                    start_time = time.time()
                                    _ = test_model_with_params(frame, temp_config)
                                    end_time = time.time()
                                    fps_values.append(1 / (end_time - start_time))

                            avg_fps = np.mean(fps_values) if fps_values else 0
                            if avg_fps > best_fps:
                                best_fps = avg_fps
                                best_params = temp_config

    if best_params:
        config.update(best_params)
        save_config()
        print(f'Parámetros óptimos: {best_params}')
        print(f'FPS promedio mejorado: {best_fps:.2f}')
    else:
        print("No se encontraron parámetros óptimos.")

# Función para evaluar rendimiento
def log_performance(start_time, end_time, log_file='performance_log.json'):
    duration = end_time - start_time
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration,
        'config': config
    }

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []

    log_data.append(log_entry)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

# Función para enviar datos por puerto serial
def send_serial_data(data):
    if serial_connection.is_open:
        serial_connection.write(data.encode())

# Función para procesar el fotograma
def process_frame(frame):
    frame = adjust_brightness_contrast(frame, brightness=config['brightness'], contrast=config['contrast'])
    if config['enable_edge_filter']:
        frame = apply_edge_filter(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        results = model(frame_tensor)

    detections = {'boxes': [], 'confidences': [], 'class_ids': []}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        keep = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), config['confidence_threshold'], config['nms_iou_threshold'])
        if len(keep) > 0:
            for i in keep.flatten():
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                cls = int(class_ids[i])

                detections['boxes'].append(boxes[i])
                detections['confidences'].append(conf)
                detections['class_ids'].append(cls)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'{model.names[cls]}: {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Enviar datos al puerto serial para que el robot persiga el objeto
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                serial_data = f'OBJECT:{mid_x},{mid_y}\n'
                send_serial_data(serial_data)

    return frame, detections

# Función principal
def main():
    load_config()  # Cargar configuración inicial

    cap = cv2.VideoCapture(0)  # Cambia 0 si utilizas una cámara diferente
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        return

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir el frame de la cámara.")
            break

        adjust_parameters(frame)  # Ajustar parámetros automáticamente
        processed_frame, detections = process_frame(frame)

        # Mostrar el resultado
        cv2.imshow('Detección de objetos', processed_frame)

        end_time = time.time()
        log_performance(start_time, end_time)  # Registrar rendimiento

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    serial_connection.close()  # Cerrar conexión serial al finalizar

if __name__ == "__main__":
    main()
