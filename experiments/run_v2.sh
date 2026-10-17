#!/bin/bash
# multi-severity extraction (GPU) then the v2 grid (CPU) into a separate results dir
cd ~/ogr_revision/revision
set -e
mkdir -p /mnt/hdd/ogr_backup
[ -f /mnt/hdd/ogr_backup/crema_av.npz ] || cp -p data/crema_av.npz /mnt/hdd/ogr_backup/
[ -f /mnt/hdd/ogr_backup/ave.npz ] || cp -p data/ave.npz /mnt/hdd/ogr_backup/
../venv/bin/python -u extract_severity.py ~/ogr_revision/data_crema /mnt/hdd/ogr_ave
echo EXTRACT_DONE
REVISION_RESULTS=/mnt/hdd/ogr_results_v2 JOBS=20 ../venv/bin/python -u bench.py --grid grid_v2.txt
echo V2_DONE
