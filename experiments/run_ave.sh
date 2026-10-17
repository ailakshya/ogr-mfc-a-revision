#!/bin/bash
# waits for the AVE download, extracts features on the GPU, runs the 10-seed grid on CPU
cd ~/ogr_revision/revision
Z=/mnt/hdd/ogr_ave/videos.zip
until [ -f "$Z" ] && [ "$(stat -c %s "$Z")" -ge 5665225063 ]; do sleep 10; done
echo DOWNLOAD_OK >> ave.log
../venv/bin/python -u extract_ave.py /mnt/hdd/ogr_ave >> ave.log 2>&1 || { echo EXTRACT_FAILED >> ave.log; exit 1; }
JOBS=20 ../venv/bin/python -u bench.py --grid grid_ave.txt >> ave.log 2>&1
echo AVE_DONE >> ave.log
