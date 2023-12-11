# Résultats des entraînements

`net.pth` est le réseau obtenu après l'entraînement.

Contenu des fichers `log.dat` :

1. Epoch
2. PSNR sur le jeu de validation
3. NRMSE sur le jeu de validation
4. SSIM sur le jeu de validation

## Nomenclature des dossiers

`p<patch size>_s<stride>_b<batch size>_e<epochs>_m<milestones>`

Les milestones sont les epochs à partir desquelles le _leaning rate_ est diminué.
