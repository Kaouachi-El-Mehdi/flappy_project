# Flappy Project — Gaming Agent

Un agent qui apprend à jouer à Flappy Bird, comparé à un agent aléatoire.

## Équipe

| Rôle | Membre |
|------|--------|
| Environnement | TODO |
| Algo | TODO |
| Entraînement + suivi | TODO |
| Restitution | TODO |

## Le jeu, et pourquoi celui-ci

Environnement : [`flappy-bird-gymnasium`](https://pypi.org/project/flappy-bird-gymnasium/), une
implémentation Gymnasium de Flappy Bird.

Pourquoi Flappy Bird :
- Espace d'action discret et minimal (2 actions), donc rapide à entraîner.
- Score net et incrémental (nombre de tuyaux passés), facile à comparer entre agents.
- L'environnement s'installe et donne un score en quelques minutes (`pip install flappy-bird-gymnasium`), largement sous la barre des 30 minutes imposée.

## Observation, action, récompense

TODO — à remplir après la première exécution de l'agent aléatoire (voir `agents/random_agent.py`)
et avant tout entraînement :
- **Observation** : TODO (dimension du vecteur d'état, ce qu'il contient)
- **Actions** : TODO (0 = ne rien faire, 1 = battre des ailes)
- **Récompense** : TODO (valeurs exactes données par l'environnement, et si on les modifie)

## Score de référence (agent aléatoire)

TODO — remplir après avoir lancé :
```
python agents/random_agent.py --episodes 30
```

| | Nombre de parties | Score moyen | Score max |
|---|---|---|---|
| Agent aléatoire | 30 | TODO | TODO |

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
python training/evaluate.py --model models/best_model.zip --episodes 30
```

| | Nombre de parties | Score moyen | Score max |
|---|---|---|---|
| Agent aléatoire | 30 | TODO | TODO |
| Agent entraîné (DQN) | 30 | TODO | TODO |

## Reproductibilité

Le même entraînement a été relancé sur plusieurs graines aléatoires (seeds) pour vérifier la
variance entre runs. Voir `docs/carnet_essais.md` pour le détail des tentatives, y compris les
ratées.

## Un échec instructif

TODO — décrire une tentative qui n'a pas marché et ce qu'elle nous a appris.

## Avec plus de temps

TODO

## Installation

```
python -m venv .venv
source .venv/Scripts/activate   # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Vidéo de présentation (10-15 min)

TODO — lien (YouTube non répertorié / Loom / autre, accessible sans compte) : TODO
