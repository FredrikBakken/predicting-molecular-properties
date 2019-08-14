import os
import gc
import sys
import pandas as pd


def select_wanted(df, wanted_rows):
    df = df[df["type"] == wanted_rows]
    return df


def memory_optimization(dfs):
    for df in dfs:
        del df
    gc.collect()


def merge_multiple(df, df_merge, atom_idx):
    df = pd.merge(df, df_merge, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    return df


def merge_single(df, df_merge):
    df = pd.merge(df, df_merge, how = 'left',
                  left_on  = ['molecule_name'],
                  right_on = ['molecule_name'])
        
    return df


def merge_custom(df, df_merge):
    df = pd.merge(df, df_merge, how = 'left',
                  left_on  = ['id', 'molecule_name'],
                  right_on = ['id', 'molecule_name'])
    
    return df


def drop_col(df, cols):
    columns = list(df.columns.values)
    
    for col in cols:
        if col in columns:
            df = df.drop(col, axis=1)
            
    return df


def generate_angles(df, df_structures):
    # Map XYZ-coordinates to the dataframe
    def map_atom_info(df, df_merge, atom_idx):
        df = pd.merge(df, df_merge.drop_duplicates(subset=['molecule_name', 'atom_index']), how = 'left',
                      left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                      right_on = ['molecule_name',  'atom_index'])

        df = df.drop('atom_index', axis=1)

        return df

    for atom_idx in [0, 1]:
        df = map_atom_info(df, df_structures, atom_idx)
        df = df.rename(columns={
            'atom': f'atom_{atom_idx}',
            'x': f'x_{atom_idx}',
            'y': f'y_{atom_idx}',
            'z': f'z_{atom_idx}'})
        
        df_structures['c_x'] = df_structures.groupby('molecule_name')['x'].transform('mean')
        df_structures['c_y'] = df_structures.groupby('molecule_name')['y'].transform('mean')
        df_structures['c_z'] = df_structures.groupby('molecule_name')['z'].transform('mean')
        df_structures['atom_n'] = df_structures.groupby('molecule_name')['atom_index'].transform('max')
    
    # Calculate initial distances
    def calculate_init_distances(df):
        df['dx'] = df['x_1'] - df['x_0']
        df['dy'] = df['y_1'] - df['y_0']
        df['dz'] = df['z_1'] - df['z_0']
        df['distance'] = (df['dx']**2 + df['dy']**2 + df['dz']**2)**(1/2)
        return df

    df = calculate_init_distances(df)
    
    # Extend the distance calculations
    def extended_distance_calculations(df):
        df_temp = df.loc[:, ["molecule_name","atom_index_0","atom_index_1","distance","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()
        df_temp_ = df_temp.copy()
        df_temp_ = df_temp_.rename(columns={
            'atom_index_0': 'atom_index_1',
            'atom_index_1': 'atom_index_0',
            'x_0': 'x_1', 'y_0': 'y_1',
            'z_0': 'z_1', 'x_1': 'x_0',
            'y_1': 'y_0', 'z_1': 'z_0'})
        
        df_temp_all = pd.concat((df_temp, df_temp_), axis=0)
        
        df_temp_all["min_distance"] = df_temp_all.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')
        df_temp_all["max_distance"] = df_temp_all.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('max')
        
        df_temp = df_temp_all[df_temp_all["min_distance"] == df_temp_all["distance"]].copy()
        df_temp = df_temp.drop(['x_0', 'y_0', 'z_0', 'min_distance'], axis=1)
        df_temp = df_temp.rename(columns={
            'atom_index_0': 'atom_index',
            'atom_index_1': 'atom_index_closest',
            'distance': 'distance_closest',
            'x_1': 'x_closest',
            'y_1': 'y_closest',
            'z_1': 'z_closest'})
        
        for atom_idx in [0, 1]:
            df = map_atom_info(df, df_temp, atom_idx)
            df = df.rename(columns={
                'atom_index_closest': f'atom_index_closest_{atom_idx}',
                'distance_closest': f'distance_closest_{atom_idx}',
                'x_closest': f'x_closest_{atom_idx}',
                'y_closest': f'y_closest_{atom_idx}',
                'z_closest': f'z_closest_{atom_idx}'})
        
        df_temp = df_temp_all[df_temp_all["max_distance"] == df_temp_all["distance"]].copy()
        df_temp = df_temp.drop(['x_0', 'y_0', 'z_0', 'max_distance'], axis=1)
        df_temp = df_temp.rename(columns={
            'atom_index_0': 'atom_index',
            'atom_index_1': 'atom_index_farthest',
            'distance': 'distance_farthest',
            'x_1': 'x_farthest',
            'y_1': 'y_farthest',
            'z_1': 'z_farthest'})
        
        for atom_idx in [0, 1]:
            df = map_atom_info(df, df_temp, atom_idx)
            df = df.rename(columns={
                'atom_index_farthest': f'atom_index_farthest_{atom_idx}',
                'distance_farthest': f'distance_farthest_{atom_idx}',
                'x_farthest': f'x_farthest_{atom_idx}',
                'y_farthest': f'y_farthest_{atom_idx}',
                'z_farthest': f'z_farthest_{atom_idx}'})
        
        return df
    
    df = extended_distance_calculations(df)
    
    # Add angle features
    def add_angles(df):
        df["distance_center0"]=((df['x_0']-df['c_x'])**2+(df['y_0']-df['c_y'])**2+(df['z_0']-df['c_z'])**2)**(1/2)
        df["distance_center1"]=((df['x_1']-df['c_x'])**2+(df['y_1']-df['c_y'])**2+(df['z_1']-df['c_z'])**2)**(1/2)
        df["distance_c0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)
        df["distance_c1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)
        df["distance_f0"]=((df['x_0']-df['x_farthest_0'])**2+(df['y_0']-df['y_farthest_0'])**2+(df['z_0']-df['z_farthest_0'])**2)**(1/2)
        df["distance_f1"]=((df['x_1']-df['x_farthest_1'])**2+(df['y_1']-df['y_farthest_1'])**2+(df['z_1']-df['z_farthest_1'])**2)**(1/2)
        df["vec_center0_x"]=(df['x_0']-df['c_x'])/(df["distance_center0"]+1e-10)
        df["vec_center0_y"]=(df['y_0']-df['c_y'])/(df["distance_center0"]+1e-10)
        df["vec_center0_z"]=(df['z_0']-df['c_z'])/(df["distance_center0"]+1e-10)
        df["vec_center1_x"]=(df['x_1']-df['c_x'])/(df["distance_center1"]+1e-10)
        df["vec_center1_y"]=(df['y_1']-df['c_y'])/(df["distance_center1"]+1e-10)
        df["vec_center1_z"]=(df['z_1']-df['c_z'])/(df["distance_center1"]+1e-10)
        df["vec_c0_x"]=(df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)
        df["vec_c0_y"]=(df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)
        df["vec_c0_z"]=(df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)
        df["vec_c1_x"]=(df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)
        df["vec_c1_y"]=(df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)
        df["vec_c1_z"]=(df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)
        df["vec_f0_x"]=(df['x_0']-df['x_farthest_0'])/(df["distance_f0"]+1e-10)
        df["vec_f0_y"]=(df['y_0']-df['y_farthest_0'])/(df["distance_f0"]+1e-10)
        df["vec_f0_z"]=(df['z_0']-df['z_farthest_0'])/(df["distance_f0"]+1e-10)
        df["vec_f1_x"]=(df['x_1']-df['x_farthest_1'])/(df["distance_f1"]+1e-10)
        df["vec_f1_y"]=(df['y_1']-df['y_farthest_1'])/(df["distance_f1"]+1e-10)
        df["vec_f1_z"]=(df['z_1']-df['z_farthest_1'])/(df["distance_f1"]+1e-10)
        df["vec_x"]=(df['x_1']-df['x_0'])/df["distance"]
        df["vec_y"]=(df['y_1']-df['y_0'])/df["distance"]
        df["vec_z"]=(df['z_1']-df['z_0'])/df["distance"]
        df["cos_c0_c1"]=df["vec_c0_x"]*df["vec_c1_x"]+df["vec_c0_y"]*df["vec_c1_y"]+df["vec_c0_z"]*df["vec_c1_z"]
        df["cos_f0_f1"]=df["vec_f0_x"]*df["vec_f1_x"]+df["vec_f0_y"]*df["vec_f1_y"]+df["vec_f0_z"]*df["vec_f1_z"]
        df["cos_center0_center1"]=df["vec_center0_x"]*df["vec_center1_x"]+df["vec_center0_y"]*df["vec_center1_y"]+df["vec_center0_z"]*df["vec_center1_z"]
        df["cos_c0"]=df["vec_c0_x"]*df["vec_x"]+df["vec_c0_y"]*df["vec_y"]+df["vec_c0_z"]*df["vec_z"]
        df["cos_c1"]=df["vec_c1_x"]*df["vec_x"]+df["vec_c1_y"]*df["vec_y"]+df["vec_c1_z"]*df["vec_z"]
        df["cos_f0"]=df["vec_f0_x"]*df["vec_x"]+df["vec_f0_y"]*df["vec_y"]+df["vec_f0_z"]*df["vec_z"]
        df["cos_f1"]=df["vec_f1_x"]*df["vec_x"]+df["vec_f1_y"]*df["vec_y"]+df["vec_f1_z"]*df["vec_z"]
        df["cos_center0"]=df["vec_center0_x"]*df["vec_x"]+df["vec_center0_y"]*df["vec_y"]+df["vec_center0_z"]*df["vec_z"]
        df["cos_center1"]=df["vec_center1_x"]*df["vec_x"]+df["vec_center1_y"]*df["vec_y"]+df["vec_center1_z"]*df["vec_z"]
        # original
        df=df.drop(['vec_c0_x','vec_c0_y','vec_c0_z','vec_c1_x','vec_c1_y','vec_c1_z',
                    'vec_f0_x','vec_f0_y','vec_f0_z','vec_f1_x','vec_f1_y','vec_f1_z',
                    'vec_center0_x','vec_center0_y','vec_center0_z','vec_center1_x','vec_center1_y','vec_center1_z',
                    'vec_x','vec_y','vec_z'], axis=1)
        # extra 
        df=df.drop(['dy', 'dx', 'dz',
                    'x_0', 'x_1', 'c_x', 'x_closest_0', 'x_closest_1', 'x_farthest_0', 'x_farthest_1',
                    'y_0', 'y_1', 'c_y', 'y_closest_0', 'y_closest_1', 'y_farthest_0', 'y_farthest_1',
                    'z_0', 'z_1', 'c_z', 'z_closest_0', 'z_closest_1', 'z_farthest_0', 'z_farthest_1'], axis=1)
        return df
    
    df = add_angles(df)
    
    return df


def loader(df, train_or_test):
    # Sorted distances
    df_sorted_distances = pd.read_csv('../input/generated/sorted_distances.csv')
    df = merge_multiple(df, df_sorted_distances, 0)
    df = merge_multiple(df, df_sorted_distances, 1)
    memory_optimization([df_sorted_distances])
    
    # Dipole moments
    df_dipole_moments = pd.read_csv(f'../input/generated/{train_or_test}_ob_dipoles_mmff44.csv')
    df_dipole_moments = df_dipole_moments.loc[:, ~df_dipole_moments.columns.str.contains('^Unnamed')]
    df_dipole_moments['size'] = (df_dipole_moments['X']**2 + df_dipole_moments['Y']**2 + df_dipole_moments['Z']**2)**(1/2)
    df = merge_single(df, df_dipole_moments)
    memory_optimization([df_dipole_moments])

    # Mulliken charges
    df_mulliken_charges = pd.read_csv(f'../input/generated/best_ob_mulliken_{train_or_test}.csv')
    df_mulliken_charges = df_mulliken_charges.loc[:, ~df_mulliken_charges.columns.str.contains('^Unnamed')]
    df = merge_multiple(df, df_mulliken_charges, 0)
    df = merge_multiple(df, df_mulliken_charges, 1)
    memory_optimization([df_mulliken_charges])
    
    # Number of atoms and max distances
    df_natoms_maxdist = pd.read_csv(f'../input/generated/natoms_maxdist.csv')
    df = merge_multiple(df, df_natoms_maxdist, 0)
    df = df.drop(['atom'], axis = 1)
    memory_optimization([df_natoms_maxdist])

    # Coupling distances
    df_coupling_distances = pd.read_csv(f'../input/generated/coupling_distances.csv')
    df = merge_custom(df, df_coupling_distances)
    memory_optimization([df_coupling_distances])
    
    df = drop_col(df, ['atom_index_0', 'atom_index_1', 'atom_index_closest_0', 'atom_index_closest_1', 
                       'atom_index_farthest_0', 'atom_index_farthest_1','X', 'Y', 'Z']) #'id',
    
    return df


def convert_df(df):
    atom_representation = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    }

    coupling_representation = {
        '1JHC': 1, '1JHN': 2, '2JHH': 3, '2JHC': 4, '2JHN': 5, '3JHH': 6, '3JHC': 7, '3JHN': 8,
    }

    df = df.replace({
        'type': coupling_representation,
        'atom_x': atom_representation,
        'atom_y': atom_representation,
        'atom_0': atom_representation,
        'atom_1': atom_representation,
    })
    df.type = df.type.astype('int64')
    df.atom_x = df.atom_x.astype('int64')
    df.atom_y = df.atom_y.astype('int64')
    df.atom_0 = df.atom_0.astype('int64')
    df.atom_1 = df.atom_1.astype('int64')
    
    df['molecule_name'] = df['molecule_name'].astype('category')
    df_cats = df.select_dtypes(['category']).columns
    df[df_cats] = df[df_cats].apply(lambda x: x.cat.codes)
    
    return df


def column_dropping(df):
    df = df.drop(['molecule_name'], axis=1)
    df = df.drop(['atom_n'], axis=1)
    
    # Drop distances
    df = drop_col(df, ['d10_x', 'd11_x', 'd12_x', 'd13_x', 'd14_x', 'd15_x', 'd16_x', 'd17_x', 'd18_x', 'd19_x', 'd20_x', 'd21_x', 'd22_x', 'd23_x', 'd24_x', 'd25_x', 'd26_x', 'd27_x', 'd28_x',
                  'd10_y', 'd11_y', 'd12_y', 'd13_y', 'd14_y', 'd15_y', 'd16_y', 'd17_y', 'd18_y', 'd19_y', 'd20_y', 'd21_y', 'd22_y', 'd23_y', 'd24_y', 'd25_y', 'd26_y', 'd27_y', 'd28_y'])
    
    # Drop types
    df = drop_col(df, ['t10_x', 't11_x', 't12_x', 't13_x', 't14_x', 't15_x', 't16_x', 't17_x', 't18_x', 't19_x', 't20_x', 't21_x', 't22_x', 't23_x', 't24_x', 't25_x', 't26_x', 't27_x', 't28_x',
                  't10_y', 't11_y', 't12_y', 't13_y', 't14_y', 't15_y', 't16_y', 't17_y', 't18_y', 't19_y', 't20_y', 't21_y', 't22_y', 't23_y', 't24_y', 't25_y', 't26_y', 't27_y', 't28_y'])
    
    df = drop_col(df, ['atom_0', 'atom_1', 'eem_x', 'eem_y', 'X', 'Y', 'Z'])
    
    return df



def get_features(coupling_type):
    def load_train_test():
        train_file = pd.read_csv(f'../input/features/{coupling_type}_train.csv')
        test_file  = pd.read_csv(f'../input/features/{coupling_type}_test.csv')
        return train_file, test_file

    if os.path.exists('../input/'):
        if (os.path.exists('../input/features/') and os.path.exists(f'../input/features/{coupling_type}_train.csv') and os.path.exists(f'../input/features/{coupling_type}_test.csv')):
            return load_train_test()
        else:
            if not os.path.exists('../input/features/'):
                os.mkdir('../input/features/')
            
            # Generate training data
            df_structures = pd.read_csv('../input/structures.csv')
            df_train = pd.read_csv('../input/train.csv')
            df_train = select_wanted(df_train, coupling_type)
            df_train = generate_angles(df_train, df_structures)
            df_train = loader(df_train, 'train')
            df_train = convert_df(df_train)
            df_train = column_dropping(df_train)
            df_train.to_csv(f'../input/features/{coupling_type}_train.csv')
            df_train = None
            df_structures = None

            # Generate test data
            df_structures = pd.read_csv('../input/structures.csv')
            df_test = pd.read_csv('../input/test.csv')
            df_test = select_wanted(df_test, coupling_type)
            df_test = generate_angles(df_test, df_structures)
            df_test = loader(df_test, 'test')
            df_test = convert_df(df_test)
            df_test = column_dropping(df_test)
            df_test.to_csv(f'../input/features/{coupling_type}_test.csv')
            df_test = None
            df_structures = None

            return load_train_test()
    else:
        sys.exit(f'The ../input directory does not exist. Make sure to execute kaggle_download.py before running this script.')


# Local testing
if __name__ == '__main__':
    train_file, test_file = get_features('2JHN')

    print(train_file.head())
    print(test_file.head())
