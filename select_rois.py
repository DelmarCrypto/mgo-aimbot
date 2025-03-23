import cv2
import numpy as np
import json

# Variables globales pour la sélection de la région
drawing = False
ix, iy = -1, -1
rect = (0, 0, 0, 0)

def draw_rectangle(event, x, y, flags, param):
    """Callback pour dessiner un rectangle en cliquant et en glissant."""
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        img_copy = param.copy()
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
        cv2.imshow('Image', img_copy)

def select_roi(image_path, label):
    """Charge une image et permet de sélectionner une région d'intérêt."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur : Impossible de charger l'image '{image_path}'.")
        return None

    print(f"Sélectionnez la région pour '{label}' en cliquant et en glissant.")
    print("Appuyez sur 'S' pour sauvegarder, 'R' pour réinitialiser, ou 'Q' pour quitter.")

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle, img)

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if rect[2] > 0 and rect[3] > 0:
                print(f"Région sélectionnée pour '{label}' : {rect}")
                return rect
            else:
                print("Aucune région sélectionnée. Veuillez sélectionner une région avant de sauvegarder.")

        elif key == ord('r'):
            rect = (0, 0, 0, 0)
            img = cv2.imread(image_path)
            print("Sélection réinitialisée.")

        elif key == ord('q'):
            print("Opération annulée.")
            return None

    cv2.destroyAllWindows()

# Sélectionner les ROI pour chaque capture d'écran
rois = {}

# ROI pour la tête
head_roi = select_roi('head_screenshot.png', 'tête')
if head_roi:
    rois['head'] = head_roi

# ROI pour l'arme M4 Custom
m4_roi = select_roi('m4_screenshot.png', 'M4 Custom')
if m4_roi:
    rois['m4'] = m4_roi

# ROI pour la Pistola Mk. 2
mk2_roi = select_roi('mk2_screenshot.png', 'Pistola Mk. 2')
if mk2_roi:
    rois['mk2'] = mk2_roi

# Sauvegarder les ROI dans un fichier JSON
if rois:
    with open('rois.json', 'w') as f:
        json.dump(rois, f, indent=4)
    print("ROI sauvegardés dans 'rois.json'.")
else:
    print("Aucune ROI sauvegardée.")
