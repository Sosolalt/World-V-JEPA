# Mini V-JEPA — Plan de Réalisation

> Objectif : Construire un projet GitHub impressionnant pour être sélectionné au hackathon "Hack the World(s)" (June 19-20, 2026). Le jury évalue GitHub + LinkedIn.

---

## 1. Vue d'ensemble

Implémenter une version allégée de V-JEPA qui apprend à prédire la dynamique d'un billard 2D en espace latent (pas en pixels). Démontrer la compréhension du paradigme JEPA et la capacité à livrer un projet propre.

### Repo : `mini-vjepa`

### Narrative
"Un seul frame de billard ne contient que des positions. Pour prédire le futur, il faut comprendre les vitesses — ce qui nécessite un contexte temporel. Notre modèle encode plusieurs frames, prédit le futur en espace latent, et apprend la physique sans supervision."

---

## 2. Architecture

```
                    ┌─────────────────────────────────┐
                    │         CONTEXT (4 frames)       │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Context Encoder f_θ (×4)      │
                    │   CNN + Spatial Self-Attention   │
                    │   → [B, 64, 128] per frame      │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Temporal Aggregation           │
                    │   4×64 = 256 tokens              │
                    │   + temporal position embeddings │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Predictor g_φ                  │
                    │   Transformer (3 layers, 4 heads)│
                    │   + time token (Δt embedding)    │
                    │   → [B, 64, 128] predicted       │
                    └────────────────┬────────────────┘
                                     │
                              MSE + VarReg Loss
                                     │
                    ┌────────────────▼────────────────┐
                    │   Target Encoder f_θ̄ (EMA)      │
                    │   Même architecture que f_θ      │
                    │   τ: 0.996 → 1.0 (cosine)       │
                    │   → [B, 64, 128] target          │
                    │   (stop-gradient)                │
                    └─────────────────────────────────┘
```

### 2.1 Context Encoder

```python
# CNN backbone (spatial feature extraction)
Conv2d(3, 64, 3, stride=2, padding=1)   # 64×64 → 32×32
GroupNorm(32, 64) + GELU
Conv2d(64, 128, 3, stride=2, padding=1)  # 32×32 → 16×16
GroupNorm(32, 128) + GELU
Conv2d(128, 256, 3, stride=2, padding=1) # 16×16 → 8×8
GroupNorm(32, 256) + GELU
Conv2d(256, 128, 1)                      # projection → [B, 128, 8, 8]
LayerNorm([128, 8, 8])

# Flatten to tokens: [B, 64, 128]
# + 1 layer self-attention (spatial relationships between patches)
```

### 2.2 Temporal Aggregation

- Encode 4 frames indépendamment → 4 × [B, 64, 128]
- Stack → [B, 256, 128]
- Add learned temporal position embeddings (4 positions × 64 spatial)
- Option simple : causal temporal attention within each spatial position

### 2.3 Predictor (Transformer)

```python
# Input: 256 context tokens + 1 time token (Δt)
# 3 layers, 4 heads, dim=128, FFN=256
# Pre-norm (LayerNorm), GELU, no dropout
# Output: 64 predicted tokens (= predicted spatial representation of future frame)
```

### 2.4 Target Encoder (EMA)

- Copie exacte du context encoder
- Poids NON entraînés par backprop
- Mise à jour : `θ_target ← τ·θ_target + (1-τ)·θ_online`
- Schedule cosine : τ de 0.996 → 1.0

### 2.5 Loss

```python
# Prediction loss
mse_loss = F.mse_loss(z_pred, z_target.detach())

# Variance regularization (anti-collapse)
std = z_pred.std(dim=0)
var_loss = F.relu(1.0 - std).mean()

# Total
loss = mse_loss + 1.0 * var_loss
```

### 2.6 Paramètres

| Paramètre | Valeur |
|-----------|--------|
| latent_dim | 128 |
| spatial_tokens | 64 (8×8) |
| context_frames | 4 |
| predictor_layers | 3 |
| predictor_heads | 4 |
| Δt range | [1, 12] (random per sample) |
| Total params | ~5-8M |

---

## 3. Données : Simulation Billard 2D

### 3.1 Physique (pymunk)

| Paramètre | Valeur |
|-----------|--------|
| Table | ratio 2:1 (normalisé à 1.0 × 0.5 en coordonnées internes) |
| Nb billes | 9 |
| Rayon bille | 0.03 (unités normalisées) |
| Masse | 0.170 kg |
| Élasticité ball-ball | 0.92 |
| Friction ball-ball | 0.05 |
| Élasticité cushion | 0.75 |
| Friction cushion | 0.15 |
| Rolling friction | 0.005 (explicite, force opposée à vélocité) |
| Velocity threshold | 0.001 (snap to zero below) |
| Physics FPS | 120 Hz |
| Poches | NON (dynamique continue) |
| Spin/rotation | NON (invisible à 64×64) |

### 3.2 Conditions Initiales (diversité)

| Stratégie | Poids | Description |
|-----------|-------|-------------|
| break | 30% | Triangle rack + cue ball frappé (3-6 m/s) |
| midgame_cluster | 20% | 3-5 billes groupées, 1 en mouvement |
| midgame_spread | 20% | Billes réparties, 1-3 en mouvement |
| two_ball | 15% | Collision simple 2 billes (patterns faciles) |
| random_velocities | 15% | Positions random, 40% des billes en mouvement |

### 3.3 Rendu

- OpenCV headless (cv2.circle), pas de pygame
- 64×64 RGB
- Fond vert (34, 139, 34), bordure marron (1-2 px)
- Billes : 9 couleurs distinctes saturées (blanc, rouge, bleu, vert, orange, violet, jaune, marron, noir)
- Rayon visuel : 3-4 px

### 3.4 Dataset

| Config | Billes | Séquences | Usage |
|--------|--------|-----------|-------|
| simple | 2-3 | 2000 | Debug + montrer scaling |
| default | 9 | 10000 | Training principal |
| stress | 15 | 1000 | Montrer limites + généralisation |

- Format : NPZ compressé
- Frames : `(N, 32, 64, 64, 3)` uint8
- Metadata : `positions (N, 32, 9, 2)` float32, `velocities (N, 32, 9, 2)` float32
- Taille estimée : ~2 GB (uint8)
- Temps génération : ~3-5 min sur M3

### 3.5 Validation Physique

- Plot énergie cinétique totale au cours du temps (décroissance monotone avec friction)
- Vérification loi de réflexion aux collisions
- Conservation du momentum lors des collisions

---

## 4. Training

### 4.1 Setup

| Paramètre | Valeur |
|-----------|--------|
| Hardware | MacBook Pro M3 32GB, PyTorch MPS |
| Precision | float32 (pas de fp16 sur MPS) |
| Batch size | 64 |
| Optimizer | AdamW |
| LR | 1.5e-4 peak |
| LR schedule | Warmup 10 epochs + cosine decay → 1e-5 |
| Weight decay | 0.05 (encoder), 0.0 (predictor) |
| Grad clip | 1.0 |
| EMA τ | 0.996 → 1.0 (cosine schedule) |
| Epochs | 150 |
| Temps estimé | ~2h |
| num_workers | 0 (data en RAM) |
| pin_memory | False |
| torch.compile | Non |

### 4.2 Training Loop

```python
for epoch in range(150):
    for batch in dataloader:  # (B, 32, 3, 64, 64) uint8
        frames = batch.float() / 255.0
        
        # Random context window and prediction target
        t_start = random.randint(0, 16)
        context = frames[:, t_start:t_start+4]     # 4 context frames
        delta_t = random.randint(1, 12)
        target_frame = frames[:, t_start + 3 + delta_t]
        
        # Forward: encode context (4 frames → aggregated tokens)
        context_tokens = encode_and_aggregate(encoder, context)
        
        # Forward: predict future
        z_pred = predictor(context_tokens, delta_t)
        
        # Forward: target (no grad)
        with torch.no_grad():
            z_target = target_encoder(target_frame)  # [B, 64, 128]
        
        # Loss
        loss = mse_loss(z_pred, z_target) + var_reg(z_pred)
        
        # Backward
        loss.backward()
        clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        
        # EMA update
        update_ema(encoder, target_encoder, tau)
```

### 4.3 Monitoring Anti-Collapse

Chaque epoch, calculer :
- `avg_std` : std par dimension, moyennée (sain > 0.1)
- `effective_rank` : exp(entropy des eigenvalues de la covariance) (sain > 40 pour dim=128)
- `avg_cosine_sim` : similarité moyenne entre paires (collapse si > 0.9)

Alerte si `avg_std < 0.05` pendant 5 epochs → augmenter λ_var ou baisser LR.

### 4.4 Checkpoints

- Sauvegarder tous les 25 epochs
- Garder le best par effective_rank (pas par loss — loss basse peut = collapse)
- Max 4 checkpoints (~30 MB chacun)

### 4.5 Gotchas MPS

- `torch.mps.empty_cache()` tous les 20 epochs
- `torch.mps.synchronize()` avant `.item()` si timing
- `.contiguous()` avant les convolutions si erreurs
- `PYTORCH_ENABLE_MPS_FALLBACK=1` en env var

---

## 5. Baseline : Pixel Prediction

Même encoder CNN, mais avec un décodeur qui reconstruit le frame futur en pixels.

```python
# Decoder (transposed convolutions, miroir de l'encoder)
# Loss: MSE sur pixels
# Même budget params, même nombre d'epochs
```

### Ce qu'on montre :
- Pixel baseline a un meilleur pixel-MSE (normal — optimisé pour ça)
- MAIS V-JEPA a un meilleur R² sur probing linéaire (positions billes)
- JEPA capture la physique, pixel baseline gaspille de la capacité sur le visuel

---

## 6. Évaluation

### 6.1 Métriques Quantitatives

| Métrique | Ce que ça prouve |
|----------|-----------------|
| Linear probe R² (positions) | Latent encode les positions |
| Linear probe R² (vitesses) | Latent encode les dynamiques |
| Cosine sim vs horizon | Dégradation gracieuse = vraie prédiction |
| Comparaison vs copy-last-frame | On bat le baseline trivial |
| Comparaison vs pixel baseline | JEPA > génératif sur downstream |

### 6.2 Visualisations (priorité)

1. **Trajectoire latente en PCA 2D** — point se déplace, kink aux collisions
2. **Courbe dégradation** — cosine sim vs horizon (notre modèle / copy-last / random)
3. **t-SNE coloré par vitesse** — gradient smooth = espace structuré
4. **Training curves** — loss + effective rank + variance (1 figure composite)
5. **Comparaison JEPA vs pixel** — 2×2 grid (latent PCA + probe accuracy)

### 6.3 Format

- Figures inlinées dans le README (PNG)
- 1 notebook (`notebooks/evaluation.ipynb`) qui génère tout de façon reproductible
- 1 petite table récapitulative dans le README

### 6.4 Ablation (mode secondaire)

Masking spatial sur le target frame (style V-JEPA) :
- Masquer 60-75% des tokens spatiaux du target
- Loss uniquement sur positions masquées
- Comparer : converge plus vite (tâche plus facile) mais test moins ambitieux

---

## 7. Structure du Repo

```
mini-vjepa/
├── configs/
│   ├── default.yaml          # 9 billes, config principale
│   ├── simple.yaml           # 2-3 billes, debug
│   └── stress.yaml           # 15 billes, stress test
├── mini_vjepa/
│   ├── __init__.py
│   ├── encoder.py            # CNN + spatial self-attention
│   ├── predictor.py          # Transformer predictor
│   ├── ema.py                # EMA target encoder
│   ├── masking.py            # Spatial masking (ablation)
│   ├── losses.py             # MSE + variance regularization
│   ├── dataset.py            # PyTorch Dataset, chargement
│   └── vjepa.py              # Assembly du système complet
├── simulation/
│   ├── physics.py            # Moteur physique pur (pymunk)
│   ├── renderer.py           # Rendu → numpy array (stateless)
│   └── generator.py          # Orchestration : sim + render + save
├── scripts/
│   ├── generate_data.py      # CLI : génération dataset
│   ├── train.py              # CLI : entraînement
│   └── evaluate.py           # CLI : évaluation + figures
├── baselines/
│   └── pixel_predictor.py    # Baseline pixel-MSE
├── notebooks/
│   └── evaluation.ipynb      # Résultats reproductibles
├── tests/
│   ├── test_physics.py       # Conservation énergie, loi réflexion
│   └── test_masking.py       # Masking dimensions correctes
├── assets/
│   ├── architecture.png      # Schéma (draw.io / excalidraw)
│   ├── simulation.gif        # GIF billard
│   └── results/              # Figures de résultats
├── data/
│   └── sample/               # 16 séquences pré-générées (git)
├── requirements.txt          # Versions pinnées
├── pyproject.toml            # Packaging
├── README.md
├── DESIGN.md                 # Décisions de design + ce qui a raté
└── LICENSE                   # MIT
```

---

## 8. README Structure

```markdown
# Mini V-JEPA: Learning World Models for 2D Physics

> A from-scratch implementation of V-JEPA that learns to predict billiard 
> dynamics in latent space — no pixel reconstruction, no generative decoder.

[Architecture diagram]
[GIF: simulation + latent trajectory]

## Key Results
- Linear probe R² for ball positions: X.XX
- Prediction horizon: Xframes before degradation
- JEPA vs pixel baseline: +X% on downstream probing

## Why JEPA, Not Generative?
[3 paragraphes : le cœur du projet]

## Quick Start
git clone / pip install / python train.py / python evaluate.py

## Architecture
[Texte + renvoi au schéma]

## Results
[4-5 figures inlinées avec 1-2 phrases d'interprétation chacune]

## Comparison with Pixel Prediction
[La figure 2×2 qui prouve la thèse]

## Limitations & Future Work
[3-4 bullets → "What I'd Build at the Hackathon"]

## Design Decisions
[Lien vers DESIGN.md]

## References
[4-6 papiers clés]

## License
MIT
```

---

## 9. LinkedIn

**Post (2-3 jours avant deadline) :**
- < 150 mots
- Technique, pas enthousiaste
- Mentionner V-JEPA, latent prediction, physics understanding
- Lien GitHub
- Pas de "j'espère être sélectionné" — écrire comme quelqu'un qui fait déjà ce travail
- Tags : #WorldModels #JEPA #SelfSupervisedLearning

**Profil :**
- Headline : inclure "World Models" ou "Self-Supervised Learning"
- Featured : pin le repo avec l'architecture diagram comme thumbnail

---

## 10. Timeline

| Jour | Focus | Livrable | Commit(s) |
|------|-------|----------|-----------|
| **J1** | Setup + simulation + validation physique | Repo structuré, `simulation/` complet, GIF, plots physique | `feat: project skeleton` / `feat: billiard physics simulation` / `feat: physics validation` |
| **J2** | Encoder + EMA + temporal aggregation | `mini_vjepa/encoder.py`, `ema.py` | `feat: spatial encoder with self-attention` / `feat: EMA target encoder` |
| **J3** | Predictor + training loop + loss | Training end-to-end qui tourne | `feat: transformer predictor` / `feat: training loop with collapse monitoring` |
| **J4** | Training complet + baseline pixel | Modèle entraîné, baseline entraînée | `feat: pixel prediction baseline` / `docs: training curves` |
| **J5** | Évaluation complète | Probing, t-SNE, dégradation curves, comparaison | `feat: linear probing evaluation` / `feat: latent space visualization` |
| **J6** | README + schéma + GIF résultats + DESIGN.md | Repo "prêt à montrer" | `docs: architecture diagram` / `docs: complete README with results` |
| **J7** | Polish + tests + reproducibility check + LinkedIn | Ship | `test: physics unit tests` / `chore: pin requirements` / `docs: final polish` |

---

## 11. Critères de Succès

### Minimum Viable (doit être atteint) :
- [ ] Training converge, pas de collapse
- [ ] Prédiction latente meilleure que random baseline
- [ ] JEPA bat pixel baseline sur au moins 1 métrique downstream
- [ ] README avec architecture diagram + résultats

### Objectif :
- [ ] R² > 0.7 sur probing positions
- [ ] R² > 0.5 sur probing vitesses
- [ ] Courbe dégradation gracieuse sur 12 steps
- [ ] t-SNE montre structure (gradient par vitesse)
- [ ] Multi-scale (simple/default/stress) montre scaling

### Stretch :
- [ ] Masking spatial comme ablation
- [ ] Animation trajectoire latente en PCA
- [ ] Conservation d'énergie via probed velocities
- [ ] Test généralisation (train 9 billes → test 5 billes)

---

## 12. Références Clés

| Papier | Rôle |
|--------|------|
| [V-JEPA (Bardes et al., 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) | Architecture de référence |
| [V-JEPA 2 (2025)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | World model + planning |
| [I-JEPA (Assran et al., 2023)](https://github.com/facebookresearch/ijepa) | Prédécesseur image |
| [A Path Towards Autonomous Machine Intelligence (LeCun, 2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf) | Fondation philosophique JEPA |
| [LeWorldModel (2025)](https://le-wm.github.io/) | JEPA end-to-end sans EMA |
| [VICReg (Bardes et al., 2021)](https://arxiv.org/abs/2105.04906) | Régularisation anti-collapse |

---

## 13. Risques et Mitigations

| Risque | Mitigation |
|--------|-----------|
| Representation collapse | Variance regularization + monitoring + EMA schedule |
| MPS instability | PYTORCH_ENABLE_MPS_FALLBACK=1, GroupNorm pas BatchNorm |
| Training trop long | Modèle petit (~5M), batch 64, 150 epochs = 2h |
| Résultats non convaincants | Avoir la baseline pixel pour relativiser |
| Pas le temps de polish | Jours 6-7 dédiés au README/visuals |
| Probing R² trop bas | Commencer avec simple (2-3 balls) pour valider le pipeline |
