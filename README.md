# Aimbot pour Metal Gear Online 2 (MGO2)

Ce projet est un aimbot pour *Metal Gear Online 2* (MGO2) fonctionnant sur PC via l'émulateur RPCS3. Il utilise `cv2.matchTemplate` pour détecter les têtes des adversaires et ajuster automatiquement la visée, en utilisant des captures d'écran brutes.

## Fonctionnalités
- Détection des têtes des adversaires avec `cv2.matchTemplate` à partir d'une capture d'écran.
- Détection automatique des armes équipées (M4 Custom et Pistola Mk. 2) à partir de captures d'écran du HUD.
- Fenêtre de débogage affichant les détections en temps réel.

## Prérequis
- Python 3.8 ou supérieur.
- RPCS3 configuré avec MGO2 (par exemple, via SaveMGO).
- Une manette PS3 configurée en mode XInput (voir instructions dans le script).

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-utilisateur/mgo2-aimbot.git
   cd mgo2-aimbot
