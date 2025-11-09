# Data Folder Overview

This folder contains all datasets used throughout the project, organized by their processing stage and purpose, explained also [here](https://github.com/sof1a03/DSS_groupproject/blob/main/README.md).

The structure supports a clear ETL workflow:
**Raw → Cleaned → Final + Geo**, ensuring traceability and modular data processing.

### Raw
Includes the original data files as obtained from external sources (e.g., RDW, CBS, KiM).  
These files are kept unchanged to ensure reproducibility of the ETL and analysis workflows.

### Cleaned
Contains preprocessed datasets that have undergone standardization, filtering, and basic cleaning.  
This stage includes tasks such as type correction, null-value handling, and harmonization of column names.

### Final
Holds the final processed datasets ready for analysis and visualization.  
These files are used as the main input for the dashboard and feature generation modules.

### Geo
Includes geospatial data used to map postal codes (PC4/PC6) and neighborhood boundaries.  
These datasets are required for generating location-based visualizations and interactive maps.
