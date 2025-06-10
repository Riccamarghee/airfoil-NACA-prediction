#!/bin/bash
#SBATCH --job-name=schilla
#SBATCH --nodes=1                        # Usa 2 nodi  # su ogni nodo, 2 socket, su ogni socket 56 core. Perciò sono 28 task per socket usando 2 core/sim 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1               # uso 1 core per clustering
#SBATCH --time=1:00:00                   # Tempo massimo per l'intero job
#SBATCH --array=2001-2509           # Esegui 2000 task, massimo 28 paralleli
#SBATCH --output=schilla_%A_%a.out            # Output per ogni task
#SBATCH --error=schilla_%A_%a.err             # Errore per ogni task
#SBATCH --account=IscrB_StocLung
#SBATCH --partition=dcgp_usr_prod# Partizione specifica


# Carica i moduli necessari
module purge
module load profile/base
#module load gcc
module load python/3.10.8--gcc--8.5.0
module load profile/eng
module load openfoam/9--intel-oneapi-mpi--2021.10.0--intel--2021.10.0
module load intel-oneapi-compilers/2023.2.1


# Accedi alla cartella corrispondente al task corrente
AIRFOILS_DIR="/leonardo_scratch/large/userexternal/rmargher/Airfoils_per_NACA"
cd $AIRFOILS_DIR

# Lista delle sottocartelle (puoi modificarlo se necessario)
SUBFOLDERS=($(ls -d */))

# Ottenere la sottocartella per l'attuale task
SIM_DIR=${SUBFOLDERS[$SLURM_ARRAY_TASK_ID-1]}

# Spostarsi nella directory della simulazione e lanciare lo script
cd $SIM_DIR
echo "Running features in $SIM_DIR on $(hostname) with $SLURM_CPUS_PER_TASK cores."
cp $HOME/TEMPLATE_AIRFOILS_SA/extract_schillaci_from_zenodo.py .
cp $HOME/TEMPLATE_AIRFOILS_SA/extract_zenodo_clusters.sh .
#sed -i 's/extract_schilla_from_zenodo\.py/extract_weight_schilla_from_zenodo\.py/g' extract_zenodo_clusters.sh
#cp $HOME/TEMPLATE_AIRFOILS_SA/clustering_mask.py .

# Eseguire lo script di simulazione
./extract_zenodo_clusters.sh
