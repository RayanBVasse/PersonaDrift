# PersonaDrift
This repository contains analysis scripts and derived data supporting the manuscript “Measuring Within-Person Variation in Written Communication Across Social Contexts.” The materials provided here enable full reproducibility of the reported tables and figures while respecting privacy constraints associated with naturalistic personal communication data

Repository Structure
/Data

Contains derived, non-identifiable outputs used in the manuscript analyses.

Key files include:

TableC1_bins.csv
Bin-level communication indicator values (14 equal message-count bins per environment).

TableC1_summary.csv
Mean and standard deviation of bin-level values across environments (Supplementary Table C1).

segments_metrics.csv
Segment-level metrics used to generate longitudinal figures.

TableA_within_dyad_variance.csv, TableB_dyad_coupling.csv,
TableC_between_dyad_contrasts.csv
Descriptive summaries reported in the Results and Supplementary sections.

Figure.png*
Final versions of figures included in the manuscript.

Raw chat logs are not included due to privacy and consent considerations. All analyses are performed on derived representations of the data.

/Scripts

Deterministic analysis scripts used to generate the tables and figures reported in the manuscript.

Key scripts include:

Run_environment_drift.py
Computes longitudinal, bin-level communication metrics across environments.

Make_table_c.py
Aggregates bin-level metrics to produce Supplementary Table C1.

Build_NRC_dyad.py
Lexicon-based feature extraction for dyadic communication analyses.

All scripts rely on fixed lexicons and rule-based processing. No large-language-model–based inference is used in the generation of reported results.

Reproducibility

All tables and figures reported in the manuscript can be regenerated using the scripts and data provided in this repository. Outputs are deterministic given the included inputs. Differences in file naming across environments reflect anonymization and platform-specific export formats.

Notes

Identifiers used in the original communication logs differ across environments (e.g., anonymized group aliases versus personal handles). These identifiers were mapped a priori to a single subject for analytic consistency, as described in the manuscript.
Raw chat logs are not included due to privacy constraints.
