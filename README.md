# Sprint M04 — Semantic Axes

Build a **semantic map**: pick terms, design two semantic axes from opposing word sets, produce one publication-quality 2D scatterplot.

> [!IMPORTANT]
> **Fork first.** Without forking, your Codespace opens on the original repo and you can't push. Fork → clone your fork → work on it.

## To get started 

1. Go [Molab notebook](https://molab.marimo.io/notebooks/nb_9zEp2dqxRbXDrrbQcieFDK). Or clone and open `assignment.py` with `uvx marimo edit --sandbox assignment.py`. This is a **worked example** on the cities dataset — not the deliverable.
2. Pick a case study, build your own pipeline.


You can choose one of the three datasets, or bring your own.

| File | Case study | N | Extra columns |
|---|---|---|---|
| `data/universities.csv` | U.S. higher-ed institutions | 157 | `type`, `region` |
| `data/sp500.csv` | S&P 500 (sample) | 203 | `sector` |
| `data/chemicals.csv` | Chemicals / materials | 179 | `class` |

If you bring your own, document the source and curation process. Requirements: ≥ 100 terms, ≥ 1 categorical attribute for color/shape.

## Tasks

### 1. Two semantic axes

Each axis: 3–6 words for **+ pole**, 3–6 for **− pole**. Good axis:

- Well-separated poles (cosine distance between centroids ≥ 0.3).
- Spreads the data, not piled at midpoint.
- One-sentence interpretation.

Axes should capture different, ideally orthogonal aspects — redundant axes waste half the plot.

### 2. One scatterplot

Plot each term at `(axis1, axis2)`. Encode categorical/ordinal attributes with **color** and **shape**. Follow data-viz principles:

- Clarity: readable symbols/text and no overlapping labels. Redundantly encode with shape
- Colorblind-friendly: No green and red colors in the same plot.
- Pre-attentive attention: use color/size/position to pull the eye to your story.
- Gestalt: proximity, similarity; zero lines, quadrant annotations, text anchors help.

### 3. Observations

2–4 paragraphs in notebook or `NOTE.md` placed at the repository root:

- What separates along each axis?
- Most **surprising** point/group — what does it say about the embedding?
- What would a **third axis** capture?

## Deliverable

Your repo must have:

- Code for the figure (marimo / Jupyter / `.py`).
- Reproducible pipeline — `run.sh` / Makefile / Snakemake that regenerates the figure from scratch.
- Raw data (CSV in `data/`).
- Final figure (PNG/PDF in `figs/` folder).
- Observations inline or in `NOTE.md` in the project root.

Submit by pushing to github and posting the URL to Brightspace.

## Evaluation

| Criterion | What we look for |
|---|---|
| Atomic git history | Small focused commits, meaningful messages. Not `final`, `final2`, `final-real`. |
| Reproducible pipeline | `bash run.sh` regenerates data + figure on a fresh clone (or Snakefile or Makefile). No manual steps. |
| Documentation | Explains *why* each axis and *what* the figure shows. |
| Viz quality | Clear, separated, colorblind-friendly, deliberate. |
| Task completion | Two axes, one scatterplot, observations — all present. |

## FAQ

**Teams?** Yes. Team repo, list members in `NOTE.md`, all submit same URL.

**Embedding model?** You can use any models. Smaller is ideal in light of reproducibility. Larger OK, document the model used.

**I could not find meaningful patterns. Get Unimodal/boring axis** That's information. Try named entities over abstractions, or poles that *should* separate your data. 


## My Submission – Chemical Semantic Axes

> This submission builds on the provided assignment scaffold.  
> The section below documents my custom dataset, semantic axes, visualization, and analysis.

---

### Dataset

I utilized a dataset of 179 chemical entities spanning 17 classes, including solvents, alkanes, alcohols, aromatics, drugs, acids, bases, salts, gases, polymers, sugars, amino acids, vitamins, metals, minerals, lipids, and explosives.

Each entry consists of:
- `name`: chemical name 
- `class`: categorical grouping 

My goal was to explore how a general-purpose language model organizes chemically diverse entities within a semantic embedding space.

---

### Semantic Axes

Two semantic axes were designed to probe how the embedding space organizes chemical meaning:

- **Axis 1 (x-axis):** 
  *stable / inert (–) ↔ reactive / energetic (+)* 

- **Axis 2 (y-axis):** 
  *industrial / inorganic (–) ↔ biological / organic (+)* 

Each axis was constructed using multiple pole words to reduce noise and capture shared conceptual meaning across different linguistic expressions.

---

### Visualization

Each chemical name was embedded using a sentence transformer model and projected onto both axes. The resulting coordinates (`x`, `y`) were visualized in a 2D scatter plot, with points colored by chemical class.

This allows us to observe how different chemical categories cluster (or fail to cluster) based on semantic associations learned by the model.

---

### Observations

In relation to the clusters, the axes appear to partially entangle information within the embedding space rather than cleanly separating distinct chemical properties. There is, however, observable structure. Metals, salts, and minerals tend to cluster together, aligning with low biological association and relatively high stability/inertness. This suggests that these compounds are being represented in ways that reflect their industrial or manufactured contexts rather than purely their chemical composition.

Mid-level clusters consist largely of solvents, polymers, and some aromatics, indicating a more balanced positioning across both axes. This suggests that their representations are more context-dependent, likely influenced by how they are used across both industrial and laboratory settings. Their placement implies that the model does not strongly privilege a single defining characteristic, but instead distributes meaning across multiple contextual dimensions.

In the upper regions (positive y), sugars, amino acids, vitamins, and drugs cluster together, reflecting strong biological association. Along higher x-values, we observe compounds associated with higher reactivity or energy, such as gases and explosives. While these groupings are directionally meaningful, the overlap between categories indicates that the axes are not fully orthogonal and that the embedding space encodes correlated semantic features.

Interestingly, I think that this has some implications for my final project in the course. The embedding space does not appear to represent chemical compounds through mechanistic or structural principles (e.g., bonding, molecular geometry, or functional groups). Instead, it organizes them according to semantic and contextual associations, likely shaped by patterns in biomedical, industrial, and general discourse present in the training data. As a result, categories that are chemically distinct may still overlap if they share similar contexts of use or discussion. This also highlights a limitation of the current axes, for greater separation would require axes that better isolate independent dimensions rather than relying on semantically entangled features. Perhaps more broadly, I think that I need to spend a bit more time considering the relationship between concept and language, for it is becoming clearer to me that how concepts are used/discussed in discourse are not necessarily aligned with what is structurally understood about those concepts in an ontological sense – would more attention to linguistic structure, such as more prefix-root-suffix organization built into such datasets, offer some interesting clustering behavior?  I will keep this in mind as I continue working on the final project. 
