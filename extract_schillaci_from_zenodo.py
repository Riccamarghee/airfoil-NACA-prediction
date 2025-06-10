import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import argparse

# extract schillaci features from zenodo

# Funzione per estrarre la sezione
def extractSection(poly_data):
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 0.5)
    plane.SetNormal(0, 0, 1)

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(poly_data)
    cutter.Update()

    return cutter.GetOutput()

# Funzione per calcolare il modulo della velocità
def calculateVelocityMagnitude(Ux_array, Uy_array):
    return np.sqrt(Ux_array**2 + Uy_array**2)

# Funzione per trasferire le aree delle celle ai punti
def calculatePointAreas(section):
    # Calcola le aree delle celle e trasferiscile ai punti
    cellSizeFilter = vtk.vtkCellSizeFilter()
    cellSizeFilter.SetInputData(section)
    cellSizeFilter.ComputeAreaOn()  # Assicurarsi che calcoli l'area per le celle 2D
    cellSizeFilter.Update()

    cell_to_point_data = vtk.vtkCellDataToPointData()
    cell_to_point_data.SetInputData(cellSizeFilter.GetOutput())
    cell_to_point_data.PassCellDataOff()  # Non trasferire i dati delle celle, solo i dati aggregati
    cell_to_point_data.Update()

    point_data = cell_to_point_data.GetOutput().GetPointData()
    cell_areas_array = vtk_to_numpy(point_data.GetArray("Area"))    

    # Estrai il campo Area
    return cell_areas_array

# Funzione per calcolare le medie nelle regioni definite (ponderate per area)
def calculateRegionAverages(points, velocity_magnitude, p_array, x_lines, y_bounds, cell_areas):
    results = []

    # Coordinate dei punti
    coords = vtk_to_numpy(points.GetData())  # (x, y, z)
    velocity_data = velocity_magnitude       # Modulo della velocità
    p_data = vtk_to_numpy(p_array)           # Pressione p

    for x_target in x_lines:
        for i in range(len(y_bounds) - 1):
            y_lower, y_upper = y_bounds[i], y_bounds[i + 1]

            # Maschera per selezionare i punti nella regione
            mask = (np.isclose(coords[:, 0], x_target, atol=0.1)) & \
                   (coords[:, 1] >= y_lower) & (coords[:, 1] < y_upper)
            
            if np.any(mask):
                area_weights = cell_areas[mask]
                u_mag_avg = np.average(velocity_data[mask], weights=area_weights)
                p_avg = np.average(p_data[mask], weights=area_weights)
            else:

                mask = (np.isclose(coords[:, 0], x_target, atol=0.5)) & \
                       (coords[:, 1] >= y_lower) & (coords[:, 1] < y_upper)

                if np.any(mask):
                    area_weights = cell_areas[mask]
                    u_mag_avg = np.average(velocity_data[mask], weights=area_weights)
                    p_avg = np.average(p_data[mask], weights=area_weights)

                else:
                    u_mag_avg = 30
                    p_avg = 0

            results.append([x_target, y_lower, y_upper, u_mag_avg, p_avg])

    return results

# Parser CLI
parser = argparse.ArgumentParser(description='Extract section and compute region-averaged velocity magnitude and pressure.')
parser.add_argument('input_file', type=str, help='Path to the input VTK file.')
parser.add_argument('output_file', type=str, help='Path to the output CSV file.')
args = parser.parse_args()

# Lettura del file VTK
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(args.input_file)
reader.Update()
poly_data = reader.GetOutput()

# Estrazione della sezione
section = extractSection(poly_data)

# Definizione delle regioni
x_lines = [-1, 2, 11]  # Posizioni x delle linee verticali
y_bounds = [-500, -10, -1, -0.1, 0, 0.1, 1, 10, 500]  # Boundaries di y

# Selezione dei dati
points = section.GetPoints()
U_array_vtk = section.GetPointData().GetArray("U")  # Componenti velocità Ux
U_array = vtk_to_numpy(U_array_vtk)  
Ux_array = U_array[:,0]
Uy_array = U_array[:,1]
p_array = section.GetPointData().GetArray("p")    # Pressione p

if points is not None and Ux_array is not None and Uy_array is not None and p_array is not None:
    # Calcolo del modulo della velocità
    velocity_magnitude = calculateVelocityMagnitude(Ux_array, Uy_array)
    
    # Calcolo delle aree trasferite ai punti
    cell_areas = calculatePointAreas(section)

    # Calcolo delle medie regionali pesate
    region_averages = calculateRegionAverages(points, velocity_magnitude, p_array, x_lines, y_bounds, cell_areas)

    # Salvataggio in CSV
    header = ["x_target", "y_lower", "y_upper", "velocity_magnitude_avg", "pressure_avg"]
    np.savetxt(args.output_file, region_averages, delimiter=",", header=",".join(header), comments="")
    print(f"Medie di velocità e pressione salvate in: {args.output_file}")
else:
    print("Errore: Dati di punti, Ux, Uy o p mancanti nella sezione estratta.")

