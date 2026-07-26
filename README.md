# SERUMS Examination Dataset

This repository contains the official examinations that healthcare professionals in Peru must complete before participating in **SERUMS** (Servicio Rural y Urbano Marginal de Salud), a mandatory service program in rural and underserved areas across the country.

The examinations included in this repository have been administered since **2025**. The original PDF files were downloaded from the official source, where the correct answers are highlighted in yellow. All examinations consist of multiple-choice questions with four answer options (A-D).


## Repository Structure

### `Examenes_*`

The following directories contain the original examination files:

* `2025_I`
* `2025_II`
* `2026_I`

Each directory includes:

* The original examination PDF (correct answers are highlighted in yellow).
* A CSV file containing the extracted questions and answer options.

Each CSV file has the following structure:

* One column containing the question.
* Four columns containing the answer options (A–D).
* One column containing the correct answer.

These CSV files were generated automatically using `process_exams.py`.


### `Pooled`

The `Pooled` directory contains the combined dataset with all examinations in three formats:

* CSV
* Parquet
* Pickle

This directory also includes:

* A JSON file summarizing the pooling process.
* A log file generated during the pooled dataset creation.

These files were produced using `appending_exams.py`.


## Data Processing Pipeline

The entire extraction and pooling workflow was performed programmatically.


### `process_exams.py`

This script processes each examination PDF by:

* Identifying individual questions.
* Extracting the four multiple-choice answer options.
* Detecting the correct answer based on the yellow highlighting in the original PDF.
* The output is a CSV file for each examination.


### `appending_exams.py`

This script reads all extracted CSV files and combines them into a single master dataset. During the pooling process, it adds two metadata variables:

* `convocatoria`: identifies the examination round (e.g., `2025_I`, `2025_II`, `2026_I`).
* `area`: identifies the professional discipline (e.g., Medicine, Nursing, Nutrition, etc.).


## Quality Assurance

Although the extraction pipeline is fully automated, a manual quality-control review (human) was performed to verify that the correct answers were accurately identified.


## Methodology

The extraction and processing pipeline follows the same methodology developed for the PeruMedQA project:

https://github.com/rodrigo-carrillo/PeruMedQA/tree/main

