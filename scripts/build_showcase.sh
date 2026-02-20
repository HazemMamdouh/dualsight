#!/usr/bin/env bash
# Build the README showcase from REAL footage: VOT-RGBT (RGBT234) static surveillance cameras with
# real fog / night / traffic. download -> run -> render -> regenerate docs/assets (GIFs + charts).
set -euo pipefail
SEQS=${SEQS:-fog6 walkingnight crossroad}; DEVICE=${DEVICE:-auto}; REPORT_SEQ=${REPORT_SEQ:-fog6}
dualsight fetch-data vot-rgbt --seq "$(echo $SEQS | tr ' ' ',')" --out data/vot
A=docs/assets; mkdir -p "$A"; parts=(); cparts=()
for s in $SEQS; do
  dualsight run --thermal data/vot/$s/thermal.mp4 --rgb data/vot/$s/rgb.mp4 --out runs/real_$s --device "$DEVICE" --save-video
  dualsight render --run runs/real_$s --compare
  parts+=(runs/real_$s/render.mp4); cparts+=(runs/real_$s/compare.mp4)
done
gif() {  # $1 out, $2 width, $3 seconds-per-clip, rest = inputs
  out=$1; w=$2; sec=$3; shift 3; ins=(); fc=""; i=0
  for f in "$@"; do ins+=(-t "$sec" -i "$f"); fc+="[$i:v]scale=$w:-2,setsar=1[v$i];"; i=$((i+1)); done
  cat=""; for ((k=0;k<i;k++)); do cat+="[v$k]"; done
  ffmpeg -y -loglevel error "${ins[@]}" -filter_complex "${fc}${cat}concat=n=$i:v=1[c];[c]fps=8,split[x][y];[x]palettegen=max_colors=128:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" "$out"
}
gif "$A/dashboard.gif" 900 5 "${parts[@]}"
gif "$A/before_after.gif" 720 4 "${cparts[@]}"
cp runs/real_$REPORT_SEQ/report/{timeline,trajectories,heatmap_thermal,sensors,zones,categories}.png "$A/"
echo "showcase rebuilt from real footage (attribution: data/vot/*/SOURCE.txt)"
