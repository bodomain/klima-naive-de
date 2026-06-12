# HYRAS Data Viewer

Interactive app for exploring HYRAS gridded climate data and nearby DWD station observations.

Structure:

- `backend/`: FastAPI API, HYRAS Zarr access, DWD station lookup, and preprocessing.
- `frontend/`: Vite/React user interface.
- `start_app.sh`: starts backend and frontend together.

Start from the repository root:

```bash
./start_app.sh
```

Refresh station coordinates in an existing `data_cache/stations.db`:

```bash
python3 data_viewer/backend/preprocessing.py --station-coordinates-only
```
