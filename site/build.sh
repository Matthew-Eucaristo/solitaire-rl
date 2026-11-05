#!/usr/bin/env bash
# Assembles the GitHub Pages site into _site/ (or the directory given).
# Run from the repository root: site/build.sh [outdir]
# Needs only python3 (stdlib) and, for the recordings, ffmpeg. No model code
# runs and nothing is trained or downloaded: the page's 1000-deal field is
# built from the committed eval JSONs in results/ by site/build_data.py, which
# fails if a count disagrees with README.md / RESULTS.md. Recordings are
# converted from recordings/*.gif to MP4 + JPEG poster; without ffmpeg the
# page falls back to the GIFs themselves.
set -euo pipefail
out=${1:-_site}

rm -rf "$out"
mkdir -p "$out/rec" "$out/fig"
cp site/index.html site/style.css site/app.js site/favicon.svg site/og.png \
   site/404.html site/robots.txt site/sitemap.xml site/web-ui.jpg "$out"/
cp results/figs/leaderboard.png results/figs/training_curves.png "$out/fig/"

if command -v ffmpeg >/dev/null; then
  media=mp4
  for name in $(python3 -I site/build_data.py --list-recordings); do
    gif="recordings/$name.gif"
    ffmpeg -loglevel error -y -i "$gif" -movflags +faststart -pix_fmt yuv420p \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 26 -preset slow -an \
      "$out/rec/$name.mp4"
    # Poster: the mid-game frame, the still shown before playing and for reduced motion.
    half=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$gif" | awk '{print $1 / 2}')
    ffmpeg -loglevel error -y -ss "$half" -i "$gif" -frames:v 1 -q:v 3 "$out/rec/$name.jpg"
  done
else
  media=gif
  echo "ffmpeg not found: the page will show the GIFs instead of videos" >&2
  for name in $(python3 -I site/build_data.py --list-recordings); do
    cp "recordings/$name.gif" "$out/rec/"
  done
fi

python3 -I site/build_data.py "$out" --media "$media"
du -sh "$out"
ls -la "$out" "$out/rec"
