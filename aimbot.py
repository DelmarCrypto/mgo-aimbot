import cv2
import numpy as np
from mss import mss
import vgamepad as vg
import keyboard
import time
import json

# Initialisation
sct = mss()
gamepad = vg.VX360Gamepad()

# Charger les captures d'écran
try:
    head_m4_screenshot = cv2.imread('head_m4_screenshot.png', 0)
    if head_m4_screenshot is None:
        raise FileNotFoundError("Fichier 'head_m4_screenshot.png' non trouvé.")
    
    mk2_screenshot = cv2.imread('mk2_screenshot.png', 0)
    if mk2_screenshot is None:
        raise FileNotFoundError("Fichier 'mk2_screenshot.png' non trouvé.")
except FileNotFoundError as e:
    print(e)
    exit()

# Charger les ROI depuis le fichier JSON
try:
    with open('rois.json', 'r') as f:
        rois = json.load(f)
    head_roi = rois['head']
    m4_roi = rois['m4']
    mk2_roi = rois['mk2']
except FileNotFoundError:
    print("Erreur : Fichier 'rois.json' non trouvé. Veuillez exécuter 'select_rois.py' pour définir les ROI.")
    exit()

# Extraire les templates à partir des ROI
head_template = head_m4_screenshot[head_roi[1]:head_roi[1] + head_roi[3], head_roi[0]:head_roi[0] + head_roi[2]]
w_head, h_head = head_template.shape[::-1]

m4_template = head_m4_screenshot[m4_roi[1]:m4_roi[1] + m4_roi[3], m4_roi[0]:m4_roi[0] + m4_roi[2]]
w_m4, h_m4 = m4_template.shape[::-1]

mk2_template = mk2_screenshot[mk2_roi[1]:mk2_roi[1] + mk2_roi[3], mk2_roi[0]:mk2_roi[0] + mk2_roi[2]]
w_mk2, h_mk2 = mk2_template.shape[::-1]

# Régions de capture
region = {'top': 100, 'left': 100, 'width': 800, 'height': 600}
weapon_region = {'top': 500, 'left': 600, 'width': 200, 'height': 100}

# Seuil de détection
DETECTION_THRESHOLD = 0.5

# Centre de l’écran
SCREEN_CENTER_X = region['left'] + region['width'] // 2
SCREEN_CENTER_Y = region['top'] + region['height'] // 2

# État de l’arme
using_bullet_weapon = True

# Fenêtre de débogage
DEBUG_WINDOW = "Aimbot Debug"
cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)

# Sensibilité
SENSITIVITY = 0.6

def detect_head():
    """Détecte les têtes en utilisant cv2.matchTemplate."""
    screenshot = np.array(sct.grab(region))
    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)

    # Détection multi-échelle pour gérer les variations de taille
    scales = [0.8, 1.0, 1.2]
    best_match = None
    max_val = DETECTION_THRESHOLD
    best_w, best_h = 0, 0

    for scale in scales:
        scaled_template = cv2.resize(head_template, (0, 0), fx=scale, fy=scale)
        scaled_w, scaled_h = scaled_template.shape[::-1]
        if scaled_w > screenshot_gray.shape[1] or scaled_h > screenshot_gray.shape[0]:
            continue
        result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= DETECTION_THRESHOLD)
        if loc[0].size > 0:
            for y, x in zip(loc[0], loc[1]):
                if result[y, x] > max_val:
                    max_val = result[y, x]
                    tx = region['left'] + x + scaled_w // 2
                    ty = region['top'] + y + scaled_h // 2
                    best_match = (tx, ty)
                    best_w, best_h = scaled_w, scaled_h

    # Afficher la détection
    if best_match:
        tx, ty = best_match
        top_left = (tx - best_w // 2 - region['left'], ty - best_h // 2 - region['top'])
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
        cv2.rectangle(screenshot_rgb, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(screenshot_rgb, f"Confidence: {max_val:.2f}", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(DEBUG_WINDOW, screenshot_rgb)
    cv2.waitKey(1)

    if best_match:
        dist = ((tx - SCREEN_CENTER_X) ** 2 + (ty - SCREEN_CENTER_Y) ** 2) ** 0.5
        return best_match, dist
    return None, None

def detect_weapon():
    """Détecte l’arme équipée dans la zone HUD."""
    global using_bullet_weapon
    weapon_screenshot = np.array(sct.grab(weapon_region))
    weapon_gray = cv2.cvtColor(weapon_screenshot, cv2.COLOR_BGRA2GRAY)

    bullet_match = cv2.matchTemplate(weapon_gray, m4_template, cv2.TM_CCOEFF_NORMED)
    no_bullet_match = cv2.matchTemplate(weapon_gray, mk2_template, cv2.TM_CCOEFF_NORMED)

    bullet_max = np.max(bullet_match)
    no_bullet_max = np.max(no_bullet_match)

    if bullet_max >= DETECTION_THRESHOLD and bullet_max > no_bullet_max:
        if not using_bullet_weapon:
            using_bullet_weapon = True
            print("Arme détectée : M4 (arme à balles)")
    elif no_bullet_max >= DETECTION_THRESHOLD and no_bullet_max > bullet_max:
        if using_bullet_weapon:
            using_bullet_weapon = False
            print("Arme détectée : Mk. II (sans balles)")
    else:
        print(f"Détection d’arme incertaine (M4: {bullet_max:.2f}, Mk. II: {no_bullet_max:.2f})")

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

def toggle_weapon_state():
    """Bascule manuelle avec 'T' si détection automatique échoue."""
    global using_bullet_weapon
    if keyboard.is_pressed('t'):
        using_bullet_weapon = not using_bullet_weapon
        state = "activé (arme à balles)" if using_bullet_weapon else "désactivé (arme sans balles)"
        print(f"Aimbot {state}")
        time.sleep(0.2)

def calibrate_regions():
    """Permet à l’utilisateur de calibrer les régions de capture."""
    print("Calibration des régions de capture...")
    print("1. Placez la fenêtre de RPCS3 en mode fenêtré.")
    print("2. Positionnez votre curseur sur le coin supérieur gauche de la zone de jeu, puis appuyez sur 'c'.")
    while not keyboard.is_pressed('c'):
        time.sleep(0.1)
    game_region_pos1 = keyboard._mouse.get_position()
    print(f"Coin supérieur gauche capturé : {game_region_pos1}")
    time.sleep(0.5)

    print("3. Positionnez votre curseur sur le coin inférieur droit de la zone de jeu, puis appuyez sur 'c'.")
    while not keyboard.is_pressed('c'):
        time.sleep(0.1)
    game_region_pos2 = keyboard._mouse.get_position()
    print(f"Coin inférieur droit capturé : {game_region_pos2}")
    time.sleep(0.5)

    print("4. Positionnez votre curseur sur le coin supérieur gauche de la zone HUD (icône d’arme), puis appuyez sur 'c'.")
    while not keyboard.is_pressed('c'):
        time.sleep(0.1)
    weapon_region_pos1 = keyboard._mouse.get_position()
    print(f"Coin supérieur gauche HUD capturé : {weapon_region_pos1}")
    time.sleep(0.5)

    print("5. Positionnez votre curseur sur le coin inférieur droit de la zone HUD, puis appuyez sur 'c'.")
    while not keyboard.is_pressed('c'):
        time.sleep(0.1)
    weapon_region_pos2 = keyboard._mouse.get_position()
    print(f"Coin inférieur droit HUD capturé : {weapon_region_pos2}")

    region['top'] = game_region_pos1[1]
    region['left'] = game_region_pos1[0]
    region['width'] = game_region_pos2[0] - game_region_pos1[0]
    region['height'] = game_region_pos2[1] - game_region_pos1[1]

    weapon_region['top'] = weapon_region_pos1[1]
    weapon_region['left'] = weapon_region_pos1[0]
    weapon_region['width'] = weapon_region_pos2[0] - weapon_region_pos1[0]
    weapon_region['height'] = weapon_region_pos2[1] - weapon_region_pos1[1]

    global SCREEN_CENTER_X, SCREEN_CENTER_Y
    SCREEN_CENTER_X = region['left'] + region['width'] // 2
    SCREEN_CENTER_Y = region['top'] + region['height'] // 2

    print("Calibration terminée.")
    print(f"Région de jeu : {region}")
    print(f"Région HUD : {weapon_region}")

def main():
    print("Aimbot pour MGO2 démarré.")
    print("Appuyez sur 'T' pour basculer l’arme (arme à balles/sans balles).")
    print("Appuyez sur 'Q' pour quitter.")
    print("Assurez-vous que RPCS3 est en mode fenêtré.")

    calibrate_regions()

    last_weapon_check = time.time()
    try:
        while True:
            if keyboard.is_pressed('q'):
                print("Arrêt demandé par l’utilisateur.")
                break

            toggle_weapon_state()

            current_time = time.time()
            if current_time - last_weapon_check >= 0.5:
                detect_weapon()
                last_weapon_check = current_time

            target, dist = detect_head()
            if target:
                print(f"Tête détectée à {target}, distance: {dist:.2f}")
                move_to_target(target, dist)
            else:
                move_to_target(None, None)

            time.sleep(0.016)

    except KeyboardInterrupt:
        print("Aimbot arrêté via Ctrl+C.")
    finally:
        gamepad.reset()
        gamepad.update()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        import keyboard
    except ImportError:
        print("Installez 'keyboard' avec 'pip install keyboard' (admin requis sur Windows).")
        exit()
    main()
