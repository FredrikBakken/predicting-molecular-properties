import os
import sys
import csv
import math
import pandas as pd

def distanceAB(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def distance_matrix():
    df_structures = pd.read_csv('input/structures.csv')
    mol_structures = df_structures.iloc[:, 0].unique().tolist()
    print(f'Number of unique molecules: {len(mol_structures)}')

    if os.path.exists('output/structures_v3.csv'):
        os.remove('output/structures_v3.csv')

    with open('output/structures_v3.csv', 'a', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['molecule_name', 'atom_index', 'atom', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'])
    
    molecule_distances = []

    for molecule in mol_structures:
        print(f'Calculating distances for current molecule: {molecule}')

        match = df_structures['molecule_name'].str.contains('(^' + molecule + ')')
        atoms = df_structures[match].values.tolist()

        for i in range(len(atoms)):
            current_atom = atoms[i]
            atom_distances = [molecule, current_atom[1], current_atom[2]]

            for j in range(len(atoms)):
                atom_distances.extend([distanceAB(current_atom[3], atoms[j][3], current_atom[4], atoms[j][4], current_atom[5], atoms[j][5])])

        molecule_distances.append(atom_distances)
    
    with open('output/structures_v3.csv', 'a', newline='') as csv_f:
        writer = csv.writer(csv_f)
        
        for md in molecule_distances:
            md = md + [0] * (32 - len(md))
            writer.writerow(md)


if __name__ == '__main__':
    distance_matrix()