# Raman–Ramsey numerical code and data

Python source and saved numerical records for Raman–Ramsey control design,
atomic dynamics, finite-count inference, and numerical analysis.

## Layout

- `selfcal_ramsey/`: effective-model simulations, control searches, and analysis.
- `selfcal_ramsey/publication_integrated/`: multilevel D1 simulations and analysis.
- `selfcal_ramsey/results/`: saved design and validation records.
- `selfcal_ramsey/publication_integrated/data/`: simulation records, control
  arrays, and inference lookups.
- `rb87_readout/atomic.py`: atomic dipole calculations.
- `requirements.txt`: Python dependencies.

Install dependencies with `python -m pip install -r requirements.txt`.
Run scripts from the repository root so imports between the two packages resolve.
The saved records include configuration values and random seeds. To check the
stored numerical results, run:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python selfcal_ramsey/publication_integrated/verify_release.py
```
