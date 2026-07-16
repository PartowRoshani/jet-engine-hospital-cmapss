# Dataset

This project uses the NASA C-MAPSS Turbofan Engine Degradation Dataset.

## Required Files

Place the original dataset files inside `data/raw/`.

### Stage 1 — FD001

- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

### Stage 2 — FD003

- train_FD003.txt
- test_FD003.txt
- RUL_FD003.txt

### Optional Bonus — FD004

- train_FD004.txt
- test_FD004.txt
- RUL_FD004.txt

Raw dataset files are not committed to this repository.

## Generated Directories

- `data/interim/`: labeled and temporary datasets
- `data/processed/`: engineered feature tables
- `data/splits/`: saved train, validation, and test engine IDs
