#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2: Calculating Diffusion Tensor Elements from Diffusion MRI Data

This script calculates diffusion tensor elements for each voxel in preprocessed diffusion MRI data.
It fits the diffusion tensor model to the data and saves the tensor elements for further analysis.

Usage:
    python Step2_Finding_Quantitative_Parameters.py --input_dir /path/to/processed/data --output_dir /path/to/output

Requirements:
    - Python packages: os, numpy, pandas, nibabel, scipy.linalg
"""

import os
import numpy as np
import nibabel as nib
from scipy.linalg import pinv
import pandas as pd
import argparse
from pathlib import Path


def load_data(subject_path, subject_id):
    """
    Load the 4D diffusion MRI data, bvals, and bvecs for a subject.
    
    Parameters:
    -----------
    subject_path : str
        Path to the subject's directory
    subject_id : str
        Subject identifier
        
    Returns:
    --------
    tuple
        (data, bvals, bvecs, affine) containing the 4D image data, b-values, b-vectors, and affine matrix
    """
    nifti_files = [f for f in os.listdir(subject_path) if f.endswith('.nii.gz')]
    if not nifti_files:
        raise FileNotFoundError(f"No NIFTI file found for subject {subject_id}")

    img_path = os.path.join(subject_path, nifti_files[0])
    img = nib.load(img_path)
    data = img.get_fdata()

    # Look for .bval and .bvec files or .npy files
    if os.path.exists(os.path.join(subject_path, f"{subject_id}.bval")):
        bvals = np.loadtxt(os.path.join(subject_path, f"{subject_id}.bval"))
        bvecs = np.loadtxt(os.path.join(subject_path, f"{subject_id}.bvec"))
    elif os.path.exists(os.path.join(subject_path, f"{subject_id}_bval.npy")):
        bvals = np.load(os.path.join(subject_path, f"{subject_id}_bval.npy"))
        bvecs = np.load(os.path.join(subject_path, f"{subject_id}_bvec.npy"))
    else:
        # Try to find any bval/bvec file
        bval_files = [f for f in os.listdir(subject_path) if f.endswith('.bval') or f.endswith('_bval.npy')]
        bvec_files = [f for f in os.listdir(subject_path) if f.endswith('.bvec') or f.endswith('_bvec.npy')]
        
        if not bval_files or not bvec_files:
            raise FileNotFoundError(f"No bval/bvec files found for subject {subject_id}")
        
        # Load the first file found
        if bval_files[0].endswith('.npy'):
            bvals = np.load(os.path.join(subject_path, bval_files[0]))
        else:
            bvals = np.loadtxt(os.path.join(subject_path, bval_files[0]))
            
        if bvec_files[0].endswith('.npy'):
            bvecs = np.load(os.path.join(subject_path, bvec_files[0]))
        else:
            bvecs = np.loadtxt(os.path.join(subject_path, bvec_files[0]))

    return data, bvals, bvecs, img.affine


def create_b_matrix(bvals, bvecs):
    """
    Create the B-matrix from b-values and b-vectors.
    
    Parameters:
    -----------
    bvals : ndarray
        Array of b-values
    bvecs : ndarray
        Array of b-vectors (3 x N or N x 3)
        
    Returns:
    --------
    ndarray
        B-matrix used for tensor fitting
    """
    # Ensure bvecs is in the right format (3 x N)
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T
    
    B = np.zeros((len(bvals), 6))
    for i in range(len(bvals)):
        b = bvals[i]
        x, y, z = bvecs[:, i]
        B[i] = [b * x * x, b * y * y, b * z * z, b * x * y * 2, b * x * z * 2, b * y * z * 2]
    return B


def fit_tensor(S, S0, B):
    """
    Fit the diffusion tensor using linear least squares.
    
    Parameters:
    -----------
    S : ndarray
        Signal values for all diffusion directions
    S0 : float
        Signal value for b=0
    B : ndarray
        B-matrix
        
    Returns:
    --------
    ndarray
        Fitted tensor elements [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        Y = np.log(S / S0)

    mask = np.isfinite(Y)
    if not np.any(mask):
        return np.zeros(6)

    B_masked = B[mask]
    Y_masked = Y[mask]

    try:
        D = pinv(B_masked) @ Y_masked
        return D
    except:
        return np.zeros(6)


def calculate_tensor_elements(subject_path, subject_id, output_dir=None):
    """
    Calculate tensor elements for each voxel in the brain.
    
    Parameters:
    -----------
    subject_path : str
        Path to the subject's directory
    subject_id : str
        Subject identifier
    output_dir : str, optional
        Output directory (if different from subject_path)
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if output_dir is None:
        output_dir = subject_path
    
    # Check if already processed
    tensor_path = os.path.join(output_dir, f"{subject_id}_tensor_elements.npy")
    if os.path.exists(tensor_path):
        print(f"Skipping {subject_id} (already processed)")
        return True

    print(f"Processing subject: {subject_id}")

    try:
        # Load the data
        data, bvals, bvecs, affine = load_data(subject_path, subject_id)
        print("Data loaded successfully")

        # Create the B-matrix
        B = create_b_matrix(bvals, bvecs)
        print("B-matrix created")

        # Get dimensions
        nx, ny, nz, n_dirs = data.shape
        tensor_elements = np.zeros((nx, ny, nz, 6))
        S0 = data[..., 0]  # b=0 image

        # Process each voxel
        for x in range(nx):
            if x % 10 == 0:
                print(f"Processing slice {x}/{nx} for {subject_id}")
            for y in range(ny):
                for z in range(nz):
                    if S0[x, y, z] > 0:  # Only process voxels with signal
                        S = data[x, y, z, :]
                        D = fit_tensor(S, S0[x, y, z], B)
                        tensor_elements[x, y, z, :] = D

        # Save the tensor elements
        np.save(os.path.join(output_dir, f"{subject_id}_tensor_elements.npy"), tensor_elements)

        # Save a sample tensor from the middle of the brain for verification
        # Find a valid middle voxel with a nonzero tensor
        mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
        while mid_x > 0 and np.all(tensor_elements[mid_x, mid_y, mid_z] == 0):
            mid_x -= 1  # Move up until a valid voxel is found

        sample_D = tensor_elements[mid_x, mid_y, mid_z]
        tensor_3x3 = np.array([
            [sample_D[0], sample_D[3], sample_D[4]],
            [sample_D[3], sample_D[1], sample_D[5]],
            [sample_D[4], sample_D[5], sample_D[2]]
        ])

        tensor_df = pd.DataFrame(tensor_3x3,
                               columns=['D00', 'D01', 'D02'],
                               index=['D00', 'D10', 'D20'])
        tensor_df.to_csv(os.path.join(output_dir, f"{subject_id}_sample_tensor.csv"))

        print(f"Successfully processed {subject_id}")
        print(f"Sample tensor from middle voxel:\n{tensor_3x3}")

        return True
    except Exception as e:
        print(f"Error processing {subject_id}: {str(e)}")
        return False


def process_all_subjects(base_path, output_dir=None):
    """
    Process all subjects in the dataset.
    
    Parameters:
    -----------
    base_path : str
        Path to the base directory containing subject folders
    output_dir : str, optional
        Base output directory (if different from base_path)
    """
    # Get list of subject directories
    subjects = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    if not subjects:
        print(f"No subject directories found in {base_path}")
        return

    processed_count = 0
    for subject_id in subjects:
        subject_path = os.path.join(base_path, subject_id)
        
        # Determine output directory
        if output_dir:
            subject_output_dir = os.path.join(output_dir, subject_id)
            os.makedirs(subject_output_dir, exist_ok=True)
        else:
            subject_output_dir = subject_path
        
        success = calculate_tensor_elements(subject_path, subject_id, subject_output_dir)
        if success:
            processed_count += 1

    print("\nProcessing Summary:")
    print(f"Total subjects: {len(subjects)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {len(subjects) - processed_count}")


def main():
    """Main function to parse arguments and run the processing."""
    parser = argparse.ArgumentParser(description='Calculate diffusion tensor elements from diffusion MRI data')
    parser.add_argument('--input_dir', required=True, help='Path to input directory containing preprocessed subject folders')
    parser.add_argument('--output_dir', help='Path to output directory (if different from input)')
    parser.add_argument('--subject', help='Process specific subject (optional)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Process single subject or entire dataset
    if args.subject:
        subject_path = os.path.join(args.input_dir, args.subject)
        
        if not os.path.exists(subject_path):
            print(f"Error: Subject directory {subject_path} does not exist")
            return
        
        if args.output_dir:
            subject_output_dir = os.path.join(args.output_dir, args.subject)
            os.makedirs(subject_output_dir, exist_ok=True)
        else:
            subject_output_dir = subject_path
            
        calculate_tensor_elements(subject_path, args.subject, subject_output_dir)
    else:
        process_all_subjects(args.input_dir, args.output_dir)


if __name__ == "__main__":
    # Example usage:
    # python Step2_Finding_Quantitative_Parameters.py --input_dir ../../Dataset/processed --output_dir ../../Dataset/tensors
    main() 