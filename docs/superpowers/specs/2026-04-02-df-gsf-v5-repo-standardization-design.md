# DF_GSF_v5 Repository Standardization Design

Date: 2026-04-02

## 1. Context

`DF_GSF_v5` is now the declared mainline project for the Bio-Series repository. The current directory contains a usable training pipeline, but the repository shape is not yet suitable for a public GitHub project:

- the main execution path exists, but the project boundary is unclear
- README usage examples no longer fully match the current CLI behavior
- machine-specific absolute paths are embedded in multiple YAML configs
- cache files, temporary outputs, and ad hoc artifacts are mixed into the tree
- historical scripts and thesis materials exist, but are not clearly separated from the runnable path

The goal of this work is to standardize the repository around `DF_GSF_v5` without deleting prior materials, while making the project easier to clone, configure, understand, and extend.

## 2. Goals

- Establish `DF_GSF_v5` as the clear mainline project to upload to GitHub.
- Preserve historical scripts, old model variants, and thesis materials instead of dropping them.
- Make local-machine paths configurable rather than hard-coded defaults.
- Align documentation with the current launcher, submission flow, and ablation workflow.
- Remove files that should not be versioned, such as caches and test output artifacts.
- Prepare the project for a clean git history and GitHub push.

## 3. Non-Goals

- No functional rewrite of the biological modeling pipeline.
- No redesign of model internals beyond path/config cleanup needed for repository standardization.
- No deletion of historical materials unless they are generated junk or accidental artifacts.
- No runtime acceptance in the current compute-node environment.

## 4. Repository Strategy

The repository will be organized around a simple rule:

- `DF_GSF_v5.py`, `submit_jobs.py`, `src/`, `scripts/compare_ablations.py`, `tests/`, and standard configs represent the runnable mainline.
- Older or secondary materials are preserved but clearly marked as historical or supporting assets.

### 4.1 Mainline Content

The following are considered first-class project entry points and should remain easy to discover:

- `DF_GSF_v5.py`
- `submit_jobs.py`
- `src/`
- `scripts/compare_ablations.py`
- `tests/`
- `config/` with standardized templates and documented local overrides
- top-level `README.md`

### 4.2 Preserved Historical Content

The following content stays in the repository, but will be documented as archival or supporting material rather than main execution path:

- older model files in `src/models/`
- one-off helper scripts that are not part of the normal run path
- thesis and manuscript support material in `thesis_materials/`
- design and implementation notes in `docs/`

If needed, non-mainline scripts may be moved under a clearly named archival location such as `legacy/` or `scripts/archive/`, but only when the move improves discoverability and does not create unnecessary path churn.

## 5. Configuration Standardization

The repository will adopt a two-layer configuration model.

### 5.1 Public Config Templates

The repository will include example config files that are safe for GitHub:

- example YAMLs will contain placeholders or relative guidance instead of user-specific absolute paths
- README will point users to copy an example config and fill in local paths

This becomes the documented default onboarding path.

### 5.2 Environment Variable Overrides

The code will support environment-variable overrides for key machine-specific paths so the same config can be reused across machines and clusters.

Priority order:

1. explicit CLI arguments, where available
2. environment variables
3. YAML config values

The initial override scope will cover the machine-dependent fields that currently vary by environment:

- `exp_root`
- `python_bin`
- `plink_bin`
- `gcta_bin`
- `reference_genome`
- `gtf_file`
- `pigbert_model`
- `gene2vec_model`
- dataset-specific `plink`
- dataset-specific `pheno`

The environment override design should stay simple and predictable. It should not try to build a fully dynamic templating system.

## 6. Documentation Plan

### 6.1 README Refresh

The main README will be rewritten to match the actual project shape:

- what the project is
- the current main entry points
- minimal install/dependency expectations
- config template workflow
- single-run examples
- Slurm batch submission examples
- ablation experiment usage
- result comparison workflow
- which folders are mainline versus archived/supporting materials

### 6.2 Supporting Documentation

Supporting docs will remain in the repo but be grouped more intentionally:

- `docs/` for design, plans, and technical project notes
- `thesis_materials/` for manuscript-support material
- historical scripts documented as archival utilities

The main README should link out to these instead of competing with them.

## 7. Version-Control Hygiene

The repository will gain standard ignore rules so generated artifacts are not committed accidentally.

At minimum, ignore policy should cover:

- `__pycache__/`
- `*.pyc`
- test temp outputs
- generated CSV summaries that are outputs rather than source assets
- trained model checkpoints
- result directories and Slurm logs, if they are local run artifacts

Any already tracked generated junk in the mainline tree should be removed from the working tree before the cleanup commit.

## 8. Directory Cleanup Rules

The cleanup pass should remove or stop tracking artifacts that are clearly not source material:

- Python cache directories
- temporary verification outputs under `tests/`
- accidental path-like directories such as malformed `https:` trees
- generated comparison CSVs that are outputs rather than maintained source files

The cleanup pass should preserve meaningful historical content even if old, as long as it is documentation, code, or intentional assets.

## 9. Git and Push Plan

The final repository action will be based on `DF_GSF_v5` as the mainline project, not the whole `01_benchmarks` directory.

Expected outcome:

- create or use a git work tree rooted around `DF_GSF_v5`
- stage the standardized repository structure
- create a clean commit covering the ablation feature plus repository standardization
- push to the target GitHub repository

If the current parent directory is not already a git repository, repository initialization or a clean upload workspace is acceptable as part of implementation.

## 10. Risks and Controls

### Risk 1: Breaking local execution while cleaning paths

Control:

- preserve YAML-driven behavior
- add environment overrides as a non-breaking layer
- avoid changing scientific logic

### Risk 2: Losing historical context while cleaning the tree

Control:

- preserve old materials
- move or relabel only when it clarifies boundaries
- do not delete thesis or historical model content

### Risk 3: README and code drifting again

Control:

- document only commands that are actually supported now
- use the current CLI subcommands and current ablation modes

### Risk 4: Over-scoping into a full repo rewrite

Control:

- keep focus on standardization, parameterization, and discoverability
- avoid architectural rewrites

## 11. Acceptance Criteria

This design is complete when:

- `DF_GSF_v5` is clearly the repository mainline
- main entry points are easy to identify from the root
- local-machine paths are no longer hard-coded as the only supported mechanism
- public-facing config examples are safe to share
- README matches actual project usage
- generated junk is removed or ignored
- historical scripts and materials are retained with clearer boundaries
- the project is ready for git commit and GitHub push
