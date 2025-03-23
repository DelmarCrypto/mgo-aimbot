import cv2
import numpy as np
from mss import mss
import vgamepad as vg
import time
from ultralytics import YOLO
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import threading
import os

# Initialisation de Flask et SocketIO
app = Flask(__name__, template_folder='.')  # Indiquer que les templates sont à la racine
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Initialisation des composants de l'aimbot
sct = mss()
gamepad = vg.VX360Gamepad()

# Charger le modèle YOLO
model = YOLO('best.pt')  # Assurez-vous que best.pt est dans le répertoire ou mettez à jour le chemin

# Charger les templates pour les armes
try:
    weapon_bullet_template = cv2.imread('m4_template.png', 0)
    if weapon_bullet_template is None:
        raise FileNotFoundError("Fichier 'm4_template.png' non trouvé.")
    w_weapon, h_weapon = weapon_bullet_template.shape[::-1]

    weapon_no_bullet_template = cv2.imread('mk2_template.png', 0)
    if weapon_no_bullet_template is None:
        raise FileNotFoundError("Fichier 'mk2_template.png' non trouvé.")
    w_no_bullet, h_no_bullet = weapon_no_bullet_template.shape[::-1]
except FileNotFoundError as e:
    print(e)
    exit()

# Régions de capture (à calibrer manuellement ou via une interface)
region = {'top': 100, 'left': 100, 'width': 800, 'height': 600}
weapon_region = {'top': 500, 'left': 600, 'width': 200, 'height': 100}

# Seuil de détection pour les armes
DETECTION_THRESHOLD = 0.7

# Centre de l’écran
SCREEN_CENTER_X = region['left'] + region['width'] // 2
SCREEN_CENTER_Y = region['top'] + region['height'] // 2

# État de l’arme et de l'aimbot
using_bullet_weapon = True
aimbot_running = False
aimbot_thread = None

# Sensibilité
SENSITIVITY = 0.6

def detect_head(screenshot_rgb):
    """Détecte les têtes avec YOLOv8 et retourne l'image annotée et la cible."""
    results = model(screenshot_rgb)

    best_match = None
    min_dist = float('inf')
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls == 0:  # Classe 'head'
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()
                if conf < 0.5:
                    continue
                tx = region['left'] + (x1 + x2) / 2
                ty = region['top'] + (y1 + y2) / 2
                dist = ((tx - SCREEN_CENTER_X) ** 2 + (ty - SCREEN_CENTER_Y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_match = (tx, ty)
                cv2.rectangle(screenshot_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(screenshot_rgb, f"Head: {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return screenshot_rgb, best_match, min_dist if best_match else None

def detect_weapon():
    """Détecte l’arme équipée dans la zone HUD."""
    global using_bullet_weapon
    weapon_screenshot = np.array(sct.grab(weapon_region))
    weapon_gray = cv2.cvtColor(weapon_screenshot, cv2.COLOR_BGRA2GRAY)

    bullet_match = cv2.matchTemplate(weapon_gray, weapon_bullet_template, cv2.TM_CCOEFF_NORMED)
    no_bullet_match = cv2.matchTemplate(weapon_gray, weapon_no_bullet_template, cv2.TM_CCOEFF_NORMED)

    bullet_max = np.max(bullet_match)
    no_bullet_max = np.max(no_bullet_match)

    if bullet_max >= DETECTION_THRESHOLD and bullet_max > no_bullet_max:
        if not using_bullet_weapon:
            using_bullet_weapon = True
            socketio.emit('log', {'message': "Arme détectée : M4 (arme à balles)"})
    elif no_bullet_max >= DETECTION_THRESHOLD and no_bullet_max > bullet_max:
        if using_bullet_weapon:
            using_bullet_weapon = False
            socketio.emit('log', {'message': "Arme détectée : Mk. II (sans balles)"})
    else:
        socketio.emit('log', {'message': f"Détection d’arme incertaine (M4: {bullet_max:.2f}, Mk. II: {no_bullet_max:.2f})"})

def move_to_target(target, dist):
    """Ajuste le joystick droit pour viser la tête si une arme à balles est équipée."""
    if target and using_bullet_weapon and dist is not None:
        target_x, target_y = target

        delta_x = (target_x - SCREEN_CENTER_X) / (region['width'] / 2)
        delta_y = (target_y - SCREEN_CENTER_Y) / (region['height'] / 2)

        sensitivity = SENSITIVITY * (1 - min(dist / 500, 0.5))
        joystick_x = max(-1.0, min(1.0, delta_x * sensitivity))
        joystick_y = max(-1.0, min(1.0, delta_y * sensitivity))

        gamepad.right_joystick_float(joystick_x, -joystick_y)
        gamepad.update()
    else:
        gamepad.right_joystick_float(0.0, 0.0)
        gamepad.update()

def aimbot_loop():
    """Boucle principale de l'aimbot."""
    global aimbot_running
    last_weapon_check = time.time()
    while aimbot_running:
        screenshot = np.array(sct.grab(region))
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Détecter les têtes
        annotated_image, target, dist = detect_head(screenshot_rgb)
        if target:
            socketio.emit('log', {'message': f"Tête détectée à {target}, distance: {dist:.2f}"})
            move_to_target(target, dist)
        else:
            move_to_target(None, None)

        # Convertir l'image annotée en base64 pour le streaming
        _, buffer = cv2.imencode('.jpg', annotated_image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': jpg_as_text})

        # Détecter l'arme toutes les 0.5 secondes
        current_time = time.time()
        if current_time - last_weapon_check >= 0.5:
            detect_weapon()
            last_weapon_check = current_time

        time.sleep(0.016)  # Environ 60 FPS

    # Réinitialiser la manette à l'arrêt
    gamepad.reset()
    gamepad.update()

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

# Événements WebSocket
@socketio.on('start_aimbot')
def start_aimbot():
    global aimbot_running, aimbot_thread
    if not aimbot_running:
        aimbot_running = True
        aimbot_thread = threading.Thread(target=aimbot_loop)
        aimbot_thread.start()
        emit('log', {'message': "Aimbot démarré."})

@socketio.on('stop_aimbot')
def stop_aimbot():
    global aimbot_running
    if aimbot_running:
        aimbot_running = False
        aimbot_thread.join()
        emit('log', {'message': "Aimbot arrêté."})

@socketio.on('toggle_weapon')
def toggle_weapon():
    global using_bullet_weapon
    using_bullet_weapon = not using_bullet_weapon
    state = "activé (arme à balles)" if using_bullet_weapon else "désactivé (arme sans balles)"
    emit('log', {'message': f"Aimbot {state}"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
