import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import os
import argparse

def get_label_from_folder_name01(folder_name):
    parts = folder_name.split('_')
    if len(parts) == 2:
        try:
            num1 = int(parts[1][0])
            num2 = int(parts[1][1])
            print(f"Primo numero: {num1}, Secondo numero: {num2}")
        except ValueError:
            return None  # Se non possiamo convertire i numeri, ritorna None
        if num1 == 0 and num2 != 0:
            return 0
        elif num1 != 0 and num2 == 0:
            return 1
        else:
            exit # esco se il difetto ce sia sopra che sotto 
    return None

def get_label_from_folder_name(folder_name):
    parts = folder_name.split('_')
    if len(parts) == 2:
        try:
            num2 = parts[1]  # La conversione a int rimuove automaticamente gli zeri iniziali
            print(f"Secondo numero: {num2}")
            return num2
        except ValueError:
            print("Errore: il secondo numero non è valido.")
            return None  # Ritorna None se non è possibile convertire il numero
    else:
        print("Errore: formato del nome della cartella non corretto.")
        return None



parser = argparse.ArgumentParser(description='Process VTK files and extract gradients.')
parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
args = parser.parse_args()

filename = args.input_file

# Nome della cartella corrente e della cartella superiore
current_folder = os.path.basename(os.getcwd())
parent_folder = os.path.dirname(os.getcwd())

# Costruisci il percorso del file CSV nella cartella superiore
output_file = 'features.npy'
label_file= 'label.npy'

# Controlla se il file CSV esiste
#if os.path.exists(output_file):
#    data = np.load(output_file)
#else:
    # Inizializza un array vuoto con dimensioni (0, 6, 12) se il file non esiste
#    data = np.empty((0, 6, 13))

my_csv = pd.read_csv(filename)

# Estrai l'etichetta dalla cartella corrente
label = get_label_from_folder_name(current_folder)
print(f"Label: {label}")
if label is None:
    print(f"Nome della cartella non valido o numeri non trovati nel nome della cartella '{current_folder}'.")
    exit()


rho = 1.225
nu = 1.48e-5
U = 30
Re = U/nu

coord_x = my_csv['Points:0'].to_numpy() 
coord_y = my_csv['Points:1'].to_numpy() 
u = my_csv['Ux'].to_numpy() 
v = my_csv['Uy'].to_numpy() 
p = my_csv['p'].to_numpy() 
nut = my_csv['nut'].to_numpy() # nu eff = nu + nut
dp_dx = my_csv['grad(p):0'].to_numpy()
dp_dy = my_csv['grad(p):1'].to_numpy()
du_dx = my_csv['Ux_grad:0'].to_numpy()
du_dy = my_csv['Ux_grad:1'].to_numpy()
d2u_dxdx = my_csv['grad_Ux_grad:0'].to_numpy() # 0: d(1)/dx, 1: d(1)/dy, 2:d(1)/dz, 
d2u_dxdy = my_csv['grad_Ux_grad:1'].to_numpy() # 3: d(2)(dx), 4: d(2)/dy con (1) = du_dx
d2u_dydy = my_csv['grad_Ux_grad:4'].to_numpy() # e (2) du_dy
dv_dx = my_csv['Uy_grad:0'].to_numpy()
dv_dy = my_csv['Uy_grad:1'].to_numpy()
d2v_dxdx = my_csv['grad_Uy_grad:0'].to_numpy()
d2v_dxdy = my_csv['grad_Uy_grad:1'].to_numpy() 
d2v_dydy = my_csv['grad_Uy_grad:4'].to_numpy()
area = my_csv['Area'].to_numpy()

adv_x = (u*du_dx + v*du_dy)
adv_y = (u*dv_dx + v*dv_dy)
viscous_x = -(nu + nut)*(d2u_dxdx + d2u_dydy)
viscous_y = -(nu + nut)*(d2v_dxdx + d2v_dydy)
pressure_x = (1/rho)*dp_dx
pressure_y = (1/rho)*dp_dy
viscous_x_lam = -(nu)*(d2u_dxdx + d2u_dydy)
viscous_y_lam = -(nu)*(d2v_dxdx + d2v_dydy)
viscous_x_turb  = -(nut)*(d2u_dxdx + d2u_dydy)
viscous_y_turb = -(nut)*(d2v_dxdx + d2v_dydy)
turbulent_x = -(adv_x + viscous_x  + pressure_x)
turbulent_y = -(adv_y + viscous_y  + pressure_y)

features = np.vstack([coord_x, coord_y, adv_x , 
                      viscous_x_lam , viscous_x_turb , 
                      pressure_x, turbulent_x,
                      adv_y, viscous_y_lam , 
                      viscous_y_turb, pressure_y, 
                      turbulent_y]).T

labels = ['x', 'y', 'adv_x', 'visc_x_laminar', 'visc_x_turbulent','p_x', 'tke_x', 
          'adv_y', 'visc_y_laminar', 'visc_y_turbulent','p_y', 'tke_y']

nc = 10  # Number of clusters
seed = 3847210123  # Standard seed for debugging/plotting
print('clustering...')
model = BayesianGaussianMixture(n_components=nc, random_state=seed, reg_covar=1e-3, 
                                weight_concentration_prior = 1, weight_concentration_prior_type='dirichlet_process', 
                                covariance_type='full') #, reg_covar=1e-4
try:
    # Adattamento del modello ai dati
    model.fit(features)
    print('Clustering completato con successo.')

    # Predizione dei cluster basati sul modello addestrato
    cluster_idx = model.predict(features)

except ValueError as e:
    # Gestione degli errori specifici
    print(f"Errore durante il fitting del modello: {e}")

except Exception as e:
    # Gestione di altre eccezioni generali
    print(f"Errore imprevisto: {e}")

print('fine clustering')
# "Predict" the clusters based on the trained model
cluster_idx = model.predict(features)

data_to_save = np.column_stack((features, cluster_idx))
np.save('naca0012_20100.npy', data_to_save)
# Ottieni il numero di cluster e features
num_clusters = len(np.unique(cluster_idx))
num_features = features.shape[1]+5 # Aggiungo u,v,p,nut,Area
print(f"Numero cluster:{num_clusters}")
print(f"numero features:{num_features}")

# Assicurarsi che num_clusters sia 6 e num_features sia 12
if num_clusters != nc or num_features != 17:
    print("Il numero di cluster o di features non corrisponde ai valori attesi.")
    exit()

# Creo un array per i dati della simulazione
simulation_data = np.zeros((1, num_clusters, num_features))
label_data = np.zeros((1, 1))

for i in range(num_clusters):
    simulation_data[0, i, 0:12] = np.mean(features[cluster_idx == i], axis=0)
    simulation_data[0, i, 12] = np.mean(u[cluster_idx == i], axis=0)
    simulation_data[0, i, 13] = np.mean(v[cluster_idx == i], axis=0)
    simulation_data[0, i, 14] = np.mean(p[cluster_idx == i], axis=0)
    simulation_data[0, i, 15] = np.mean(nut[cluster_idx == i], axis=0)
    simulation_data[0,i, 16] = np.sum(area[cluster_idx==i], axis=0)


simulation_data[0, :, 0]= model.means_[:,0]
simulation_data[0, :, 1]= model.means_[:,1]
# Combina i dati esistenti con i nuovi dati
#data_combined = np.vstack([data, simulation_data])
label_data[0,0]=label
# Salva i dati aggiornati nel file .npy
np.save(output_file, simulation_data)
np.save(label_file, label_data)
