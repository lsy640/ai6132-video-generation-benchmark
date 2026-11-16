#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --job-name=ai6132-report
#SBATCH --output=ai6132_report_%j.out
#SBATCH --error=ai6132_report_%j.err
# Compile the LaTeX report and the contribution statement with Tectonic. Submit from the repository root:
#   sbatch 08-report/build_report.sh
# The first run downloads the TeX packages it needs into ~/.cache/Tectonic.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/08-report/Review_of_Image_to_Video_Generation"
"${AI6132_TECTONIC:-$HOME/tools/tectonic/tectonic}" main.tex
# The course asks for the report to be submitted as group_XX_report.pdf.
/usr/bin/mv -f main.pdf group_11_report.pdf
# The statement of member contributions is built next to this script.
cd ..
for tex in group_*_contribution.tex; do
  if [ -e "$tex" ]; then "${AI6132_TECTONIC:-$HOME/tools/tectonic/tectonic}" "$tex"; fi
done
