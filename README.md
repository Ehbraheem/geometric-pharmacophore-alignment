# Geometric Pharmacophore Alignment

A Python implementation of a geometric pharmacophore alignment workflow for docking ligands against pharmacophore targets.

The project generates multiple 3D ligand conformers, aligns each conformer to a target pharmacophore using rigid-body transformations, evaluates the resulting poses with a pharmacophore scoring function, rejects poses with steric clashes, and writes the best-scoring pose for each target to an SDF file.

---

## Features

- Generate 3D conformers from SMILES
- Extract pharmacophore features from ligands
- Prepare pharmacophore targets from JSON
- Perform rigid-body pharmacophore alignment
- Detect steric clashes against excluded volumes
- Score candidate poses using Gaussian distance scoring
- Export best poses as SDF
- Comprehensive unit test suite using pytest

---

## Libraries Used

The implementation relies on the following libraries:

- **RDKit**
  - Molecule parsing
  - Conformer generation
  - MMFF optimization
  - Pharmacophore feature extraction
  - SDF reading/writing

- **NumPy**
  - Coordinate manipulation
  - Vectorized distance calculations

- **SciPy**
  - 3D rigid-body rotations (`scipy.spatial.transform.Rotation`)

- **pytest**
  - Unit testing

---

## Docking Algorithm

For each target:

1. Generate multiple 3D conformers for the ligand.
2. Extract ligand pharmacophore features.
3. Convert the JSON target definition into an optimized internal representation.
4. For every conformer:
   - compute candidate rigid-body alignments,
   - reject poses with steric clashes,
   - score the pose against the interaction sites.
5. Select the highest-scoring pose.
6. Apply the best transformation.
7. Write the docked ligand to the output SDF.

The pharmacophore score is computed as

```
score = Σ weight × exp(-(distance / 1.25)²)
```

where each interaction site contributes according to the closest matching ligand feature.

---

## Project Structure

```
dock.py                               Docking implementation
tests/                                Unit tests
data/targets.json                     Input pharmacophore targets
results/docked_poses.sdf              Docking results
notebook.ipynb                        Example workflow
```

---

## Requirements

- Python 3.14
- RDKit
- NumPy
- SciPy
- pytest

---

## Setup

### Using uv (recommended)

Install Python 3.14 if necessary:

```bash
uv python install 3.14
```

Create a virtual environment:

```bash
uv venv --python 3.14
```

Activate it:

macOS/Linux

```bash
source .venv/bin/activate
```

Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
uv sync
```

---

## Running the Notebook

Open the notebook and execute all cells.

The notebook will:

- load the pharmacophore targets,
- dock each ligand,
- export the best poses,
- visualize the resulting molecules.

---

## Running the Tests

Run the full test suite:

```bash
python -m pytest
```

or

```bash
pytest
```

---

## Output

Successful execution produces an SDF containing the best-scoring pose for each target.

During execution the notebook also reports information such as

```
=== target_3 ===
Best score:      4.392
Best conformer:  13
Rotation:
[[...]]
Translation:
[...]
```

allowing the selected pose to be inspected.
