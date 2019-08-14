import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Ref: https://www.kaggle.com/borisdee/predicting-mulliken-charges-with-acsf-descriptors?scriptVersionId=15809975

def append_to_csv(row):
    with open('./../../input/generated/dm_g1-descriptor.csv', 'a', newline='') as sm:
        wr = csv.writer(sm)
        wr.writerow(row)


def g_two():
    # Definition of the cutoff function
    def fc(Rij, Rc):
        y_1 = 0.5 * (np.cos(np.pi * Rij[Rij <= Rc] / Rc) + 1)
        y_2 = Rij[Rij > Rc] * 0
        y = np.concatenate((y_1, y_2))
        return y

    # Definition of the G2 function
    def get_G2(Rij, eta, Rs, Rc):
        return np.exp(-eta*(Rij-Rs)**2) * fc(Rij, Rc)
    
    # Open the distance matrix file
    with open('./distance_matrix.csv', 'r') as dm:
        csv_reader = csv.reader(dm, delimiter=',')

        molecule = None
        slice_index = None
        current_row = 0

        first_row = True
        for row in csv_reader:
            current_row += 1

            if not first_row:
                distances = row[3:]
                distances = [float(i) for i in distances]

                # Locate slice index
                current_molecule = row[0]
                if current_molecule != molecule:
                    print(current_molecule)
                    # sys.exit()
                    slice_index = None

                    for idx, idx_val in enumerate(reversed(distances)):
                        if idx_val != float(0):
                            slice_index = (28 - idx)
                            break

                    molecule = current_molecule
            else: 
                first_row = False


def g_one():
    # Definition of the cutoff function
    def fc(Rij, Rc):
        y_1 = 0.5 * (np.cos(np.pi * Rij[Rij <= Rc] / Rc) + 1)
        y_2 = Rij[Rij > Rc] * 0
        y = np.concatenate((y_1, y_2))
        return y
    

    # Open the distance matrix file
    with open('./../../input/generated/distance_matrix.csv', 'r') as dm:
        csv_reader = csv.reader(dm, delimiter=',')

        molecule = None
        slice_index = None
        current_row = 0

        first_row = True
        for row in csv_reader:
            current_row += 1

            if not first_row and current_row > 0:
                distances = row[3:]
                distances = [float(i) for i in distances]

                # Locate slice index
                current_molecule = row[0]
                if current_molecule != molecule:
                    print(current_molecule)
                    # sys.exit()
                    slice_index = None

                    for idx, idx_val in enumerate(reversed(distances)):
                        if idx_val != float(0):
                            slice_index = (28 - idx)
                            break

                    molecule = current_molecule
                
                distances = distances[:slice_index]

                # Convert list to numpy array
                distances = np.array(distances, dtype=np.float32)

                # Calculate G1 descriptors
                G1_01 = fc(distances, 0.5).sum()
                G1_02 = fc(distances, 1.0).sum()
                G1_03 = fc(distances, 1.5).sum()
                G1_04 = fc(distances, 2.0).sum()
                G1_05 = fc(distances, 2.5).sum()
                G1_06 = fc(distances, 3.0).sum()
                G1_07 = fc(distances, 3.5).sum()
                G1_08 = fc(distances, 4.0).sum()
                G1_09 = fc(distances, 4.5).sum()
                G1_10 = fc(distances, 5.0).sum() # 1JHN
                G1_11 = fc(distances, 5.5).sum()
                G1_12 = fc(distances, 6.0).sum()
                G1_13 = fc(distances, 6.5).sum()
                G1_14 = fc(distances, 7.0).sum()

                desc_row = row[:3] + [G1_01, G1_02, G1_03, G1_04, G1_05, G1_06, G1_07, G1_08, G1_09, G1_10, G1_11, G1_12, G1_13, G1_14]
                append_to_csv(desc_row)
                desc_row = []
            else:
                first_row = False


if __name__ == '__main__':
    g_one()
    # g_two()
