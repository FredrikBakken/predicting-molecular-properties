import sys
import pandas as pd


def gen_coup_dist():
    df_structures = pd.read_csv('./input/structures.csv')
    molecules = df_structures.iloc[:, 0].unique().tolist()

    df_train = pd.read_csv('./input/train.csv')
    train_molecules = df_train.iloc[:, 1].unique().tolist()

    df_test = pd.read_csv('./input/test.csv')
    # test_molecules = df_test.iloc[:, 1].unique().tolist()

    # Loop through all molecules
    for molecule in molecules:
        print(molecule)

        current_df = None

        if molecule in train_molecules:
            # Found in train
            current_df = df_train.loc[df_train['molecule_name'].isin([molecule])]
        else:
            # Found in test
            current_df = df_test.loc[df_test['molecule_name'].isin([molecule])]
        
        print(current_df)

        atom_indices_0 = current_df.iloc[:, 2].tolist()
        atom_indices_1 = current_df.iloc[:, 3].tolist()
        print(atom_indices_0, atom_indices_1)

        

        sys.exit()


if __name__ == '__main__':
    gen_coup_dist()
