import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import argparse

# extract 2d section in z=0.5 from CFD simulations and compute gradients.

def extractSection(poly_data):
    # Creating the cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 0.5)
    plane.SetNormal(0, 0, 1) #Orthogonal to the xy plane

    # Cutting the space in the first direction
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(poly_data)
    cutter.Update()

    # Extracting the first target section
    target_section = cutter.GetOutput()

    return target_section

parser = argparse.ArgumentParser(description='Process VTK files and extract gradients.')
parser.add_argument('input_file', type=str, help='Path to the input VTK file.')
parser.add_argument('output_file', type=str, help='Path to the output CSV file.')
args = parser.parse_args()

file_path = args.input_file
fileOut = args.output_file

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(file_path)
reader.Update()
poly_data = reader.GetOutput()

# Gradiente di Ux
gradient_Ux = vtk.vtkGradientFilter()
gradient_Ux.SetInputData(poly_data)
gradient_Ux.SetInputScalars(0, "Ux")
gradient_Ux.SetResultArrayName("Ux_grad")
gradient_Ux.Update()
poly_data.GetPointData().AddArray(gradient_Ux.GetOutput().GetPointData().GetArray("Ux_grad"))
#Ux_grad_array = vtk_to_numpy(poly_data.GetPointData().GetArray("Ux_grad"))

# Gradiente di Uy
gradient_Uy = vtk.vtkGradientFilter()
gradient_Uy.SetInputData(poly_data)
gradient_Uy.SetInputScalars(0, "Uy")
gradient_Uy.SetResultArrayName("Uy_grad")
gradient_Uy.Update()
poly_data.GetPointData().AddArray(gradient_Uy.GetOutput().GetPointData().GetArray("Uy_grad"))

#Gradiente di grad(Ux) dUx_x/dx, dU_x/dy, dU_x/dz, dU_y/dx, dU_y/dy, dU_y/dz, dU_z/dx, dU_z/dy, dU_z/dz
gradient_Ux_grad = vtk.vtkGradientFilter()
gradient_Ux_grad.SetInputData(poly_data)
gradient_Ux_grad.SetInputScalars(0, "Ux_grad")
gradient_Ux_grad.SetResultArrayName("grad_Ux_grad")
gradient_Ux_grad.Update()
poly_data.GetPointData().AddArray(gradient_Ux_grad.GetOutput().GetPointData().GetArray("grad_Ux_grad"))

# Gradiente di grad(Uy)
gradient_Uy_grad = vtk.vtkGradientFilter()
gradient_Uy_grad.SetInputData(poly_data)
gradient_Uy_grad.SetInputScalars(0, "Uy_grad")
gradient_Uy_grad.SetResultArrayName("grad_Uy_grad")
gradient_Uy_grad.Update()
poly_data.GetPointData().AddArray(gradient_Uy_grad.GetOutput().GetPointData().GetArray("grad_Uy_grad"))

section= extractSection(poly_data)

clip = vtk.vtkBoxClipDataSet()
clip.SetInputDataObject(section)
clip.SetBoxClip(-4.0, 10.0, -2.5, 2.5, 0, 1)
clip.Update()
box = clip.GetOutput()

# Da provare!!
cellSizeFilter = vtk.vtkCellSizeFilter()
cellSizeFilter.SetInputData(box)
cellSizeFilter.ComputeAreaOn()  # Assicurarsi che calcoli l'area per le celle 2D
cellSizeFilter.Update()

cell_to_point_data = vtk.vtkCellDataToPointData()
cell_to_point_data.SetInputData(cellSizeFilter.GetOutput())
cell_to_point_data.PassCellDataOff()  # Non trasferire i dati delle celle, solo i dati aggregati
cell_to_point_data.Update()

point_data = cell_to_point_data.GetOutput().GetPointData()
cell_areas_array = vtk_to_numpy(point_data.GetArray("Area"))

points = box.GetPoints()
table = vtk.vtkAttributeDataToTableFilter()
table.SetInputData(box)
table.Update()
table.GetOutput().AddColumn(points.GetData())
table.Update()
table.GetOutput().AddColumn(point_data.GetArray("Area")) 
table.Update()
writer = vtk.vtkDelimitedTextWriter()
writer.SetInputConnection(table.GetOutputPort())
writer.SetFileName(fileOut)
writer.Update()
writer.Write()

print(f"Lo script è andato a buon fine. File salvato come: {fileOut}")

