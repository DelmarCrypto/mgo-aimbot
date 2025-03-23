# Aimbot pour Metal Gear Online 2 (MGO2) avec YOLOv8

Ce projet est un aimbot pour *Metal Gear Online 2* (MGO2) fonctionnant sur PC via l'émulateur RPCS3. Il utilise YOLOv8 pour détecter les têtes des adversaires et une interface web (Flask) pour contrôler l'aimbot en temps réel.

## Fonctionnalités
- Détection des têtes des adversaires avec YOLOv8 (robuste aux variations comme les bérets, têtes nues, etc.).
- Détection automatique des armes équipées (M4 Custom et Pistola Mk. 2).
- Interface web pour démarrer/arrêter l'aimbot, basculer les modes d'arme, et voir un flux vidéo en temps réel.
- Fenêtre de débogage streamée dans le navigateur avec les détections annotées.

## Prérequis
- Python 3.8 ou supérieur.
- RPCS3 configuré avec MGO2 (par exemple, via SaveMGO).
- Un modèle YOLOv8 entraîné pour détecter les têtes dans MGO2 (voir section "Entraînement de YOLOv8").
- Une manette PS3 configurée en mode XInput (voir instructions dans le script).

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-utilisateur/mgo2-aimbot.git
   cd mgo2-aimbot
