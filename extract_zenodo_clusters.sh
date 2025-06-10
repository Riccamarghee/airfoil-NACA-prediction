# Imposta il numero di processi per job 
#nProcs=$SLURM_CPUS_PER_TASK

# Carica i moduli necessari
#module purge
#module load profile/base
#module load gcc
#module load python/3.10.8--gcc--8.5.0
#module load profile/eng
#module load openfoam/9--intel-oneapi-mpi--2021.10.0--intel--2021.10.0
#module load intel-oneapi-compilers/2023.2.1

# compute clusters from zenodo. set clustering or clustering_prop to change strategy

echo "Job started at $(date)" > JOBTIME_schilla.info
#currentFolder=$(basename "$PWD")
#VtkFile="${currentFolder}.vtk"
echo "Controllo esistenza del file: $VtkFile"
#echo "$VtkFile"
FileDir="../no_vtk"


if [ ! -d "VTK" ]; then
    echo "La cartella VTK non esiste. Scrivo il nome della directory corrente in $FileDir."
    
    
    # Aggiungi il nome della directory corrente al file no_vtk solo se non è già presente
    if ! grep -Fxq "$currentFolder" "$FileDir"; then
        echo "$currentFolder" >> "$FileDir"
    else
        echo "Il nome della directory corrente è già presente in $FileDir."
    fi
    exit 1
else
# Entra nella cartella VTK e controlla se il file esiste
	Time_late=$(foamListTimes -latestTime)
	currentFolder=$(basename "$PWD")
	VtkFile="${currentFolder}_${Time_late}.vtk"
	if [ -f "VTK/$VtkFile" ]; then
    		echo "File $VtkFile esistente."
	else
    		echo "File $VtkFile non trovato. Lanciando reconstructPar al time precedente."
    
    	# Trova il time precedente a LatestTime
    		rm -r processor*/$Time_late
    		rm -r $Time_late

    		# Lancia reconstructPar e foamToVTK per il time precedente
    		reconstructPar -latestTime > log.reco
    		# Post-processamento
    		simpleFoam -postProcess -func "grad(p)" -latestTime
    		simpleFoam -postProcess -func R -latestTime
    		simpleFoam -postProcess -func "components(U)" -latestTime
    		Time_late=$(foamListTimes -latestTime)
    		foamToVTK 
    		VtkFile="${currentFolder}_${Time_late}.vtk"
    		echo "Reconstrution and VTK conversion completed for time $Time_late."
	fi
fi
# estraggo sezione e calcolo gradienti
SectionFile="${currentFolder}_schilla.csv"
python extract_schillaci_from_zenodo.py "VTK/$VtkFile" $SectionFile >> log.schilla 
echo "Job finished at $(date)" >> JOBTIME_schilla.info

#Clustering
python clustering_prop.py $SectionFile >> log.clustering


echo "Job finished at $(date)" >> JOBTIME_schilla.info





