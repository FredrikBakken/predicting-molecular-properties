import os
import sys


def check_directories():
    directories = [
        ['./input'],
        ['./input/features'],
        ['./input/generated'],
        ['./input/zipped_source'],
        ['./models'],
        ['./models/nn'],
        ['./models/xgb'],
        ['./notebooks'],
        ['./submissions'],
        ['./utils'],
        ['./utils/other'],
    ]

    for directory in directories:
        if os.path.exists(directory[0]):
            directory.append('exists')
            directory.append('')
        else:
            if './input' in directory[0]:
                directory.append('missing')
                directory.append(' - Please run the ./utils/kaggle_download.py script.')
            elif './models' in directory[0] or './submissions' in directory[0]:
                directory.append('created')
                directory.append(' - Missing directory created.')
                os.mkdir(directory[0])
            elif './notebooks'in directory[0] or './utils' in directory[0]:
                directory.append('missing')
                directory.append(' - Please clone/pull files from the Github repository.')
    
    return directories


def check_files():
    files = [
        ['./input/dipole_moments.csv'],
        ['./input/magnetic_shielding_tensors.csv'],
        ['./input/mulliken_charges.csv'],
        ['./input/potential_energy.csv'],
        ['./input/sample_submission.csv'],
        ['./input/scalar_coupling_contributions.csv'],
        ['./input/structures.csv'],
        ['./input/structures.zip'],
        ['./input/test.csv'],
        ['./input/train.csv'],
        ['./input/generated/best_ob_mulliken_test.csv'],
        ['./input/generated/best_ob_mulliken_train.csv'],
        ['./input/generated/coupling_distances.csv'],
        ['./input/generated/distance_matrix.csv'],
        ['./input/generated/natoms_maxdist.csv'],
        ['./input/generated/sorted_distances.csv'],
        ['./input/generated/test_ob_dipoles_mmff44.csv'],
        ['./input/generated/train_ob_dipoles_mmff44.csv'],
        ['./input/generated/dm_g1-descriptor.csv'],
        ['./notebooks/main.ipynb'],
        ['./submissions/submission_best.csv'],
        ['./utils/check_repository.py'],
        ['./utils/generate_features.py'],
        ['./utils/kaggle_download.py'],
        ['./utils/other/distance_matrix.py'],
        ['./utils/other/dm_descriptors.py'],
    ]

    for cfile in files:
        if os.path.exists(cfile[0]):
            cfile.append('exists')
        else:
            cfile.append('missing')
    
    return files


def check_repository():
    os.chdir('./../')
    
    # Check that all directories exists
    directories = check_directories()

    # Check that all files exists
    files = check_files()

    # Print results
    print('\n**  Directories:')
    [print(f'({line[1]}) {line[0]} {line[2]}') for line in directories]
    
    print('\n**  Files:')
    [print(f'({line[1]}) {line[0]}') for line in files]


if __name__ == '__main__':
    check_repository()
