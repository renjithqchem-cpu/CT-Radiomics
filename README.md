# CT radiomics and low energy electron attachment: code and data for radiosensitizer prioritization

Code and derived data accompanying the manuscript "Can CT radiomics inform radiosensitizer prioritization? A low energy electron attachment screening study" (Renjith Bhaskaran, submitted to Biomedical Physics & Engineering Express).

This repository implements a transparent computational framework that bridges two normally separate domains, CT-based tumor radiomics and molecular dissociative electron attachment (DEA) physics, to ask what it would take for CT radiomics to inform the prioritization of candidate radiosensitizer molecules. Pretreatment CT scans from the NSCLC-Radiomics cohort (421 patients) are used to construct a two-group computational partition by unsupervised clustering, which is tested by bootstrap stability, permutation null, alternative algorithms, supervised reproducibility, acquisition (scanner and ComBat harmonization) sensitivity and clinical association analyses. Each group is weighted by a CT-derived relative electron density proxy (RED proxy) and combined with a constraint based low energy electron (LEE) distribution to score five literature verified DEA cross-section records, with Monte Carlo uncertainty propagation and a spectral requirement analysis of the resulting candidate ranking. The framework is a screening and prioritization proof of concept, not a clinical prediction model.

## Pipeline order

The complete workflow is organized into 21 steps (22 files, because step 1 uses two scripts and step 20 uses one helper module), progressing from patient-level CT and segmentation conversion through radiomics extraction, phenotype construction and robustness testing, acquisition and clinical analyses, DEA and LEE scoring, uncertainty and scenario analyses, and final technical auditing. The step numbers are those of Table 1 of the manuscript.

1. `dicom_to_nifti.py` (with `fix_tumour_masks.py`) -- converts per-patient TCIA DICOM CT series and segmentation objects into NIfTI image and mask pairs (`image.nii.gz`, `mask.nii.gz`), selecting the "Neoplasm, Primary" segment by its label and not by its position.
2. `mask_qc.py` -- checks every mask for voxel agreement with the DICOM segmentation object, slice alignment and tumor volume, and writes `mask_qc.csv`.
3. `extract_radiomics.py` -- runs PyRadiomics feature extraction (bin width 25 HU, native geometry) on the NIfTI pairs and writes a patient-level table of 107 descriptors.
4. `correlate_analysis_v4_lee_peak.py` -- applies the one-pass Spearman redundancy filter (27 retained features), median imputation, standardization and k-means (k = 2) to define the two groups, then performs RED proxy weighting, LEE-bin assignment and DEA scoring against the DEA database and LEE distribution.
5. `exact_pipeline_bootstrap.py` -- evaluates cluster stability on the fixed 27-feature representation with 1000 bootstrap resamples and clusterwise Jaccard similarity.
6. `phenotype_stability_analyses.py` (classifier module) -- reproduces the group labels from the 27 retained features with random forest and logistic regression (five-fold cross-validation, scaling inside each training fold, with and without shape features) and writes `classifier_restricted.csv`.
7. `cluster_null_permutation_test.py` -- compares the observed maximum silhouette (k = 2 to 5) with a feature permutation null of 5000 permutations and reports the (b+1)/(m+1) empirical p-value.
8. `k2_vs_k3_bootstrap_deltaS.py` -- bootstraps the difference between the k = 2 and k = 3 silhouette coefficients (1000 resamples) and reports its 95% percentile interval.
9. `feature_threshold_sensitivity.py` -- tests the sensitivity of the partition to Spearman thresholds of 0.80, 0.85, 0.90 and 0.95 (retained features, silhouette, ARI and bootstrap stability).
10. `phenotype_stability_analyses.py` -- runs the full pipeline bootstrap, matched k = 2 and k = 3 Jaccard analysis, gap statistic, Ward and Gaussian mixture alternatives and the tumor volume association.
11. `final_imaging_analyses.py` -- describes the retained features (Cohen's d, random forest importance) and performs the size-removed threshold analysis and the null distribution summary.
12. `dicom_acquisition_summary.py` -- extracts scanner model, kVp, reconstruction kernel, pixel spacing and tube current from one DICOM header per patient, tests their association with the partition and writes `acquisition_parameters.csv`.
13. `extract_radiomics_resampled.py` -- repeats the feature extraction after 1 mm isotropic resampling and compares the resulting partition with the native-geometry partition (ARI).
14. `dea_resonance_profiles.py` -- builds the source-controlled DEA candidate database (peak energy, cross section, species and channel, evidence type, provenance) and writes `DEA_verified_scoring_data.csv`.
15. `dea_uncertainty_sensitivity.py` -- propagates DEA uncertainty by Monte Carlo (10,000 simulations) to give candidate rank probabilities and pairwise comparison probabilities.
16. `dea_scenario_sensitivity.py` -- tests the ranking under flip thresholds, resonance width smoothing (up to 2 eV), admissible LEE spectra and temperature and uncertainty scenarios.
17. `final_technical_audit.py` -- performs the final integrity checks (patient and feature counts, RED proxy weighting, LEE normalization, bin assignment, score construction and DEA provenance) and writes the audit report.
18. `clinical_association.py` -- tests the association of the group labels with overall stage, T and N category, histology, sex, age and survival (chi-square, Mann-Whitney, log-rank and Cox models) using the TCIA clinical table.
19. `rtstruct_check.py` -- compares every tumor mask with the RTSTRUCT contour of the primary tumor (Dice, volumes, slice range) and writes `rtstruct_check.csv`.
20. `combat_sensitivity.py` (requires `combat.py`) -- repeats the partition analysis after ComBat harmonization for scanner model, reconstruction kernel, or scanner model with tube voltage (with and without tumor volume preserved or removed).
21. `lee_flip_analysis.py` -- finds which phenotype specific LEE spectra, if any, would reorder the five candidates (flip region and random admissible spectra).

`combat.py` is a self-contained parametric empirical-Bayes ComBat implementation (Johnson et al. 2007) used by step 20.

## Data inputs

- Imaging: NSCLC-Radiomics collection, TCIA, https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI (not redistributed here; download via the TCIA/NBIA Data Retriever).
- `DEA_verified_scoring_data.csv` and `DEA_verified_database_final.xlsx` -- the five quantitative DEA records used in the manuscript, with peak energy, peak cross section, uncertainty information and full source citation per record.
- `lee_energy_distribution.csv` -- constraint based 25-bin (1 eV) LEE probability vector anchored to the published value of Pimblott and LaVerne (2007), Radiat. Phys. Chem. 76, 1244-1247.
- Derived per-patient tables (TCIA patient identifiers and derived values only): the radiomics feature table (native and 1 mm resampled), `acquisition_parameters.csv`, `mask_qc.csv`, `rtstruct_check.csv`.
- `NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv` -- clinical table of the TCIA collection (version 3, October 2019), included for the clinical association analysis.

```

## Requirements

See `requirements.txt` and `environment.yml` (exact versions in `requirements_pinned.txt`). Developed with Python 3.10, PyRadiomics 3.0.1, SimpleITK, pydicom, scikit-image, scikit-learn 1.4.2, pandas, numpy, scipy and matplotlib (the Cox model is implemented directly in `clinical_association.py`). The random seed for every stochastic step is 20260826.

## Citation

If you use this code, please cite the associated manuscript (full citation to be added on publication), the archived release (DOI: 10.5281/zenodo.24026755) and, where relevant, the original data sources listed above.

## License

Released under the MIT License, see `LICENSE`.

## Contact

Renjith Bhaskaran, Department of Chemistry, Madanapalle Institute of Technology & Science (MITS), India. drrenjithb@mits.ac.in
