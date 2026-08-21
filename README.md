# Flappy Project — Gaming Agent

Un agent qui apprend à jouer à Flappy Bird, comparé à un agent aléatoire.

## Équipe

| Rôle | Membre |
|------|--------|
| Environnement | Mostafa |
| Algo | Saddem |
| Entraînement + suivi | El Mehdi |
| Restitution | Anass |

## Le jeu, et pourquoi celui-ci

Environnement : [`flappy-bird-gymnasium`](https://pypi.org/project/flappy-bird-gymnasium/), une
implémentation Gymnasium de Flappy Bird.

Pourquoi Flappy Bird :
- Espace d'action discret et minimal (2 actions), donc rapide à entraîner.
- Score net et incrémental (nombre de tuyaux passés), facile à comparer entre agents.
- L'environnement s'installe et donne un score en quelques minutes (`pip install flappy-bird-gymnasium`), largement sous la barre des 30 minutes imposée.

## Observation, action, récompense

Vérifié en inspectant `FlappyBird-v0` directement (`use_lidar=False`), avant tout entraînement :

- **Observation** : vecteur de 12 valeurs continues, normalisées entre -1 et 1
  (`Box(-1.0, 1.0, (12,), float64)`). Il encode la position/vitesse verticale de l'oiseau et la
  position des prochains tuyaux (variante simplifiée, sans LIDAR — 180 valeurs en mode LIDAR).
- **Actions** : `Discrete(2)` — `0` = ne rien faire (l'oiseau tombe), `1` = battre des ailes (saut).
- **Récompense** (valeurs par défaut de l'environnement, non modifiées pour la première itération) :
  - `+0.1` par frame en vie
  - `+1.0` à chaque tuyau passé (incrémente aussi `info["score"]`)
  - `-0.5` en "zone privée" (trop près du bord d'un tuyau, quasi-collision)
  - `-1.0` à la mort

## Score de référence (agent aléatoire)

```
python agents/random_agent.py --episodes 30
```

| | Nombre de parties | Score moyen | Score max |
|---|---|---|---|
| Agent aléatoire | 30 | 0.00 | 0 |

Sur 30 parties, l'agent aléatoire ne passe **aucun** tuyau : Flappy Bird ne pardonne pas les actions
aléatoires (une seule mauvaise décision suffit à mourir), donc `0` est une référence légitime et
attendue. L'agent entraîné n'a donc besoin que de passer au moins un tuyau de façon répétée pour
prouver qu'il a appris mieux que le hasard.

## Méthode d'apprentissage

**DQN** (Deep Q-Network), via [stable-baselines3](https://stable-baselines3.readthedocs.io/).

Pourquoi DQN pour ce jeu : l'espace d'action est discret et minimal (2 actions), ce qui correspond
exactement au cas d'usage de DQN. C'est aussi l'algorithme le plus simple à mettre en place et à
déboguer pour une première itération, cohérent avec la consigne "fonction de récompense simple
d'abord, méthode complexifiée seulement si le temps le permet".

## Entraînement

```
python training/train_dqn.py --seed 0 --timesteps 200000
```

Chaque run est identifié par un `--seed`. Le meilleur modèle rencontré pendant l'entraînement est
sauvegardé automatiquement (`models/checkpoints/seed_<seed>/best_model.zip`), et les scores
d'évaluation périodiques sont enregistrés dans `logs/eval_seed_<seed>/evaluations.npz` pour tracer
la courbe de progression.

## Courbe de progression

```
python training/plot_progress.py --seeds 0 1 2 --baseline-score <score_moyen_aleatoire>
```

TODO — insérer la courbe (`logs/training_curve.png`) une fois générée, et commenter ce qu'elle montre.

## Résultat final : aléatoire vs entraîné

TODO — à remplir après reload du meilleur modèle depuis un script neuf :
```
python training/evaluate_v2.py --model models/best_model.zip --episodes 30
```

|    Agent                | Nombre de parties     | Score moyen | Score max |

| Agent aléatoire         | 30                    | 0.00        | 0         |
| Agent entraîné (DQN)V2  | 30                    | 33.43       | 184       |

## Reproductibilité

Le même entraînement a été relancé sur plusieurs graines aléatoires (seeds) pour vérifier la
variance entre runs. Voir `docs/carnet_essais.md` pour le détail des tentatives, y compris les
ratées.


3. **Replace `## Un échec instructif` with this**

```markdown
## Un échec instructif

Le premier DQN entraîné pendant 200 000 timesteps obtenait seulement un score moyen de 0.47 et un maximum de 1.

Nous avons ensuite augmenté l'entraînement à 1 000 000 de timesteps, ce qui a permis d'atteindre un score moyen de 4.03 et un maximum de 11.

Un entraînement encore plus long de 2 000 000 de timesteps n'a cependant pas amélioré les résultats : le score moyen est redescendu à 3.27.

Cela nous a montré qu'augmenter uniquement la durée d'entraînement ne garantit pas une meilleure politique.

Nous avons donc créé une deuxième version du DQN avec une fonction de récompense davantage orientée vers le passage des tuyaux, un réseau plus grand et une exploration plus longue.

Cette version a obtenu un score moyen de 33.43 et un maximum de 181.

## Avec plus de temps

Avec plus de temps, nous pourrions :

- tester davantage de seeds pour mesurer plus précisément la variance ;
- comparer DQN avec PPO ;
- effectuer une recherche automatique des hyperparamètres ;
- tester l'observation LIDAR ;
- entraîner l'agent sur davantage de timesteps ;
- améliorer encore la fonction de récompense.

## Installation

```
python -m venv .venv
source .venv/Scripts/activate   # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Vidéo de présentation (10-15 min)

TODO — lien (YouTube non répertorié / Loom / autre, accessible sans compte) : TODO
