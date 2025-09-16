import h5py
import os
import sys

def create_reduced_compas_file(input_filename, output_filename):
    """
    Create a reduced COMPAS HDF5 file containing only the essential datasets
    needed for Cosmic Integration calculations.

    Parameters:
    -----------
    input_filename : str
        Path to the original COMPAS HDF5 file
    output_filename : str
        Path for the reduced output HDF5 file
    """

    with h5py.File(input_filename, 'r') as input_file, \
            h5py.File(output_filename, 'w') as output_file:

        # Copy BSE_System_Parameters group
        if 'BSE_System_Parameters' in input_file:
            sys_group = output_file.create_group('BSE_System_Parameters')
            input_sys = input_file['BSE_System_Parameters']

            # Essential system parameter datasets
            required_sys_datasets = [
                'SEED',
                'Stellar_Type@ZAMS(1)',
                'Stellar_Type@ZAMS(2)',
                'Mass@ZAMS(1)',
                'Mass@ZAMS(2)',
                'Metallicity@ZAMS(1)'
            ]

            # Optional CHE datasets
            optional_sys_datasets = [
                'CH_on_MS(1)',
                'CH_on_MS(2)'
            ]

            # Copy required datasets
            for dataset_name in required_sys_datasets:
                if dataset_name in input_sys:
                    input_sys.copy(dataset_name, sys_group)
                else:
                    print(f"Warning: Required dataset '{dataset_name}' not found in BSE_System_Parameters")

            # Copy optional datasets if available
            for dataset_name in optional_sys_datasets:
                if dataset_name in input_sys:
                    input_sys.copy(dataset_name, sys_group)

        # Copy BSE_Double_Compact_Objects group
        if 'BSE_Double_Compact_Objects' in input_file:
            dco_group = output_file.create_group('BSE_Double_Compact_Objects')
            input_dco = input_file['BSE_Double_Compact_Objects']

            # Essential DCO datasets
            required_dco_datasets = [
                'SEED',
                'Stellar_Type(1)',
                'Stellar_Type(2)',
                'Mass(1)',
                'Mass(2)',
                'Time',
                'Coalescence_Time',
                'Merges_Hubble_Time'
            ]

            # Copy required datasets
            for dataset_name in required_dco_datasets:
                if dataset_name in input_dco:
                    input_dco.copy(dataset_name, dco_group)
                else:
                    print(f"Warning: Required dataset '{dataset_name}' not found in BSE_Double_Compact_Objects")

        # Copy BSE_Common_Envelopes group (with optional datasets)
        if 'BSE_Common_Envelopes' in input_file:
            cee_group = output_file.create_group('BSE_Common_Envelopes')
            input_cee = input_file['BSE_Common_Envelopes']

            # Common envelope datasets (all optional based on your code)
            optional_cee_datasets = [
                'SEED',
                'Immediate_RLOF>CE',
                'Optimistic_CE'
            ]

            # Copy datasets if available
            for dataset_name in optional_cee_datasets:
                if dataset_name in input_cee:
                    input_cee.copy(dataset_name, cee_group)

    print(f"Reduced COMPAS file created: {output_filename}")

    # Print file size comparison

    original_size = os.path.getsize(input_filename) / (1024 ** 2)  # MB
    reduced_size = os.path.getsize(output_filename) / (1024 ** 2)  # MB
    compression_ratio = (1 - reduced_size / original_size) * 100

    print(f"Original file size: {original_size:.2f} MB")
    print(f"Reduced file size: {reduced_size:.2f} MB")
    print(f"Size reduction: {compression_ratio:.1f}%")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python reduce_compas_datafile.py <input_compas_file> <output_reduced_file>")
        sys.exit(1)

    input_compas_file = sys.argv[1]
    output_reduced_file = sys.argv[2]

    create_reduced_compas_file(input_compas_file, output_reduced_file)