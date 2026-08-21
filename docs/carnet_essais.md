# Carnet d'essais

## Essai 1 — DQN V1, 200 000 timesteps

Seed : 0

Score moyen : 0.47  
Score maximum : 1

Conclusion : l'agent apprend légèrement, mais les performances restent très faibles.

## Essai 2 — DQN V1, 1 000 000 timesteps

Seed : 3

Score moyen : 4.03  
Score maximum : 11

Conclusion : augmenter la durée d'entraînement améliore fortement les performances.

## Essai 3 — DQN V1, 2 000 000 timesteps

Seed : 4

Score moyen : 3.27  
Score maximum : 10

Conclusion : doubler encore la durée d'entraînement n'améliore pas automatiquement la politique.

## Essai 4 — DQN V2, 2 000 000 timesteps

Seed : 5

Modifications :

- reward shaping ;
- récompense plus importante lors du passage d'un tuyau ;
- réseau de neurones `[256, 256]` ;
- buffer de replay augmenté à 200 000 ;
- batch size de 128 ;
- phase d'exploration plus longue ;
- `learning_starts` augmenté à 10 000.

Résultats :

Score moyen : 33.43  
Score maximum : 181

Conclusion : la modification de la fonction de récompense et des hyperparamètres a produit une amélioration majeure.