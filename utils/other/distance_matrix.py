import os
import sys
import csv
import math
import pandas as pd

def distanceAB(x1, x2, y1, y2, z1, z2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def distance_matrix():
    ''' Distance Matrix
    Calculate the distance matrix for each molecule in the dataset.
    '''
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


def distance_matrix_sorted():
    ''' Sorted Distance Matrix
    Calculate the distance matrix in sorted order, closest to farthest.
    '''
    df_distance_matrices = pd.read_csv('./input/generated/distance_matrix.csv')
    atom_representation = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    }

    def append_to_csv(row):
        with open('./output/sorted_distances_2.csv', 'a', newline='') as sm:
            wr = csv.writer(sm)
            wr.writerow(row)

    with open('./input/generated/distance_matrix.csv', 'r') as dm:
        csv_reader = csv.reader(dm, delimiter=',')

        atoms = None
        molecule = None
        molecule_atom_types = {}
        result_types = []
        results = []

        current_row = 0

        first_row = True
        for row in csv_reader:
            current_row += 1

            if not first_row and current_row >= 1794299 and current_row <= 1794319:
                current_molecule = row[0]

                # Go through data => Update atom types => Reset
                if current_molecule != molecule:
                    print(current_molecule)
                    atoms = df_distance_matrices.loc[df_distance_matrices['molecule_name'].isin([current_molecule])]
                    molecule = current_molecule
                    molecule_atom_types.clear()
                    result_distances = []
                
                molecule_atom_types[int(row[1])] = row[2]

                atom_type_dict = {}

                # FORMAT DISTANCES
                distances = row[3:]
                distances = [float(i) for i in distances]
                for idx, distance in enumerate(distances):
                    if distance != 0.0:
                        atom_type_dict[distance] = idx

                distances = sorted(distances)

                remove_to = None
                for idx, val in enumerate(reversed(distances)):
                    if val == 0.0:
                        remove_to = idx
                        break
                
                distances = distances[len(distances) - remove_to:]
                distances = sorted(distances)
                distances = distances + [0.0] * (29 - len(distances))

                # POSITIONAL INFORMATION
                result_types = []

                for distance in distances:
                    if distance != 0.0:
                        distance_position = atom_type_dict[distance]
                        result_types.append(atom_representation[atoms.loc[atoms['atom_index'] == distance_position, 'atom'].item()])


                result_types = result_types + [0] * (29 - len(result_types))

                results = row[:3]
                results.extend(distances)
                results.extend(result_types)

                append_to_csv(results)
                results = []
            else:
                first_row = False


def natoms_maxdist_coupdist():
    ''' Collection of Data Generation
    Generate the number of atoms, max distances, and coupling distances.
    '''
    def append_to_csv(row, filename):
        with open(f'./output/{filename}.csv', 'a', newline='') as outf:
            wr = csv.writer(outf)
            wr.writerow(row)
    
    with open('./input/generated/distance_matrix.csv', 'r') as dm:
        csv_reader = csv.reader(dm, delimiter=',')

        molecule = None
        slice_index = None
        current_row = 0

        first_row = True
        for row in csv_reader:
            current_row += 1
            print(current_row)

            if not first_row:
                distances = row[3:]
                distances = [float(i) for i in distances]

                # Locate slice index
                current_molecule = row[0]
                if current_molecule != molecule:
                    print(current_molecule)
                    slice_index = None

                    for idx, idx_val in enumerate(reversed(distances)):
                        if idx_val != float(0):
                            slice_index = (28 - idx)
                            break

                    molecule = current_molecule
                
                distances = distances[:slice_index]

                print(distances)
                print(f'Number of atoms: {(slice_index)}')
                print(f'Max distance: {max(distances)}')
                out_row = row[:3] + [slice_index + 1] + [max(distances)]
                print(out_row)

                append_to_csv(out_row, 'natoms_max-dist')
            else:
                first_row = False



if __name__ == '__main__':
    distance_matrix()
    distance_matrix_sorted()
    natoms_maxdist_coupdist()
