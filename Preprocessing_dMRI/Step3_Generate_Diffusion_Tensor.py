#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4: Generate Diffusion Tensor Parameters

This script calculates diffusion tensor parameters (FA, MD, AD, RD) from preprocessed diffusion MRI data.
It also organizes images by their 4th dimension (number of diffusion directions) and saves the tensor parameters.

Usage:
    python Step4_Generate_Diffusion_Tensor.py --subject_path /path/to/subject --output_base /path/to/output

Requirements:
    - Python packages: dipy, nibabel, numpy, scipy
"""

import os
import shutil
import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path


def copy_nifti_with_dimension_check(subjects_directory, output_base_directory, verbose=True):
    """
    Copies 4D NIfTI images to different directories based on the 4th dimension size.
    
    Parameters:
    -----------
    subjects_directory : str
        Path to the directory containing subject folders
    output_base_directory : str
        Path to the base output directory
    verbose : bool, optional
        Whether to print detailed information
    
    Returns:
    --------
    dict
        Dictionary with dimension categories and their file lists
    """
    dimension_directories = {
        46: os.path.join(output_base_directory, 'all_46_diff_4d_img_ordered'),
        54: os.path.join(output_base_directory, 'all_54_diff_4d_img_ordered'),
        55: os.path.join(output_base_directory, 'all_55_diff_4d_img_ordered')
    }
    
    dimension_files = {46: [], 54: [], 55: []}

    # Create dimension-specific directories if they don't exist
    for dim, path in dimension_directories.items():
        if not os.path.exists(path):
            os.makedirs(path)

    # Process each subject directory
    for subject_id in os.listdir(subjects_directory):
        subject_path = os.path.join(subjects_directory, subject_id)
        if not os.path.isdir(subject_path):
            continue

        nifti_files = [f for f in os.listdir(subject_path) if f.startswith("4d_img") and f.endswith(".nii.gz")]

        for nifti_file in nifti_files:
            nifti_path = os.path.join(subject_path, nifti_file)
            try:
                nii_img = nib.load(nifti_path)
                fourth_dim = nii_img.shape[-1]

                if fourth_dim in dimension_directories:
                    dest_path = os.path.join(dimension_directories[fourth_dim], nifti_file)
                    shutil.copy(nifti_path, dest_path)
                    dimension_files[fourth_dim].append(nifti_file)
                    if verbose:
                        print(f"Copied {nifti_file} to {dimension_directories[fourth_dim]}")
            except Exception as e:
                print(f"Error processing {nifti_path}: {e}")
    
    return dimension_files


def extract_subject_id_from_filename(filename):
    """
    Extract subject ID from filename.
    
    Parameters:
    -----------
    filename : str
        Filename to extract subject ID from
        
    Returns:
    --------
    str
        Subject ID
    """
    # Remove prefix and suffix
    parts = filename.replace("4d_img_DTI_", "").replace(".nii.gz", "").split("_")
    
    # Extract subject ID parts (depends on naming convention)
    # This assumes format like: 4d_img_DTI_016_S_4009_M_91_AD.nii.gz
    if len(parts) >= 3:
        return "_".join(parts)
    else:
        return filename


def get_reference_bval_bvec(dataset_path, mode=46):
    """
    Get reference bval and bvec files for a specific diffusion mode.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset
    mode : int, optional
        Number of diffusion directions (46, 54, or 55)
        
    Returns:
    --------
    tuple
        (bvals_path, bvecs_path) containing paths to reference bval and bvec files
    """
    # Reference subjects for each mode
    reference_subjects = {
        46: "016_S_4009",
        54: "027_S_6648",
        55: "011_S_4827"
    }
    
    if mode not in reference_subjects:
        raise ValueError(f"Unsupported mode: {mode}. Must be 46, 54, or 55.")
    
    subject_id = reference_subjects[mode]
    subject_path = os.path.join(dataset_path, subject_id)
    
    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Reference subject path not found: {subject_path}")
    
    bvals_path = os.path.join(subject_path, f"{subject_id}_bval.npy")
    bvecs_path = os.path.join(subject_path, f"{subject_id}_bvec.npy")
    
    if not os.path.exists(bvals_path) or not os.path.exists(bvecs_path):
        raise FileNotFoundError(f"bval/bvec files not found for reference subject {subject_id}")
    
    return bvals_path, bvecs_path


def process_subjects_by_mode(input_dir, output_base_dir, mode, dataset_path, verbose=True):
    """
    Process all subjects with a specific diffusion mode.
    
    Parameters:
    -----------
    input_dir : str
        Path to the directory containing NIfTI files for this mode
    output_base_dir : str
        Path to the base output directory
    mode : int
        Number of diffusion directions (46, 54, or 55)
    dataset_path : str
        Path to the dataset containing reference bval/bvec files
    verbose : bool, optional
        Whether to print detailed information
    """
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    # Get reference bval and bvec files
    try:
        bvals_path, bvecs_path = get_reference_bval_bvec(dataset_path, mode)
    except Exception as e:
        print(f"Error getting reference bval/bvec files: {e}")
        return
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, f"DTI_parameters_{mode}_diff")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load gradient table once
    bvals = np.load(bvals_path)
    bvecs = np.load(bvecs_path)
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    
    if verbose:
        print(f"Processing subjects with {mode} diffusion directions")
        print(gtab.info)
    
    # Process each NIfTI file
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            subject_id = extract_subject_id_from_filename(filename)
            
            if verbose:
                print(f"Processing subject: {subject_id}")
            
            # Create subject output directory
            subject_output_dir = os.path.join(output_dir, subject_id)
            os.makedirs(subject_output_dir, exist_ok=True)
            
            # Load image
            filepath = os.path.join(input_dir, filename)
            img = nib.load(filepath)
            data = img.get_fdata()
            affine = img.affine
            
            if verbose:
                print(f"Image shape: {data.shape}")
            
            # Create brain mask
            maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, 
                                        numpass=2, autocrop=False, dilate=1)
            
            # Apply Gaussian smoothing
            fwhm = 1.25
            gauss_std = fwhm / np.sqrt(8 * np.log(2))  # Convert FWHM to Gaussian std (~0.53)
            data_smooth = np.zeros(data.shape)
            
            for v in range(data.shape[-1]):
                data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
            
            # Fit tensor model
            tenmodel = dti.TensorModel(gtab)
            tenfit = tenmodel.fit(maskdata, mask=mask)
            
            # Get tensor values
            tensor_D = tenfit.quadratic_form  # 3x3 diffusion tensor matrix
            tensor_val = dti.lower_triangular(tensor_D)  # 6 lower triangular values [Dxx,Dxy,Dyy,Dxz,Dyz,Dzz]
            
            # Calculate quantitative parameters
            D_est = dti.from_lower_triangular(tensor_val)
            eigvals, eigvecs = dti.decompose_tensor(D_est, min_diffusivity=0)
            tensor_fa = dti.fractional_anisotropy(eigvals)
            tensor_md = dti.mean_diffusivity(eigvals)
            tensor_rd = dti.radial_diffusivity(eigvals)
            tensor_ad = dti.axial_diffusivity(eigvals, axis=-1)
            
            # Save tensor values
            for i in range(6):
                saved_image = nib.Nifti1Image(tensor_val[..., i], img.affine)
                nib.save(saved_image, os.path.join(subject_output_dir, 
                                                f'diffusion_tensor_val_{i}_{subject_id}.nii.gz'))
            
            # Save quantitative parameters
            saved_image_fa = nib.Nifti1Image(tensor_fa, img.affine)
            nib.save(saved_image_fa, os.path.join(subject_output_dir, f'tensor_fa_{subject_id}.nii.gz'))
            
            saved_image_md = nib.Nifti1Image(tensor_md, img.affine)
            nib.save(saved_image_md, os.path.join(subject_output_dir, f'tensor_md_{subject_id}.nii.gz'))
            
            saved_image_ad = nib.Nifti1Image(tensor_ad, img.affine)
            nib.save(saved_image_ad, os.path.join(subject_output_dir, f'tensor_ad_{subject_id}.nii.gz'))
            
            saved_image_rd = nib.Nifti1Image(tensor_rd, img.affine)
            nib.save(saved_image_rd, os.path.join(subject_output_dir, f'tensor_rd_{subject_id}.nii.gz'))
            
            saved_image_mask = nib.Nifti1Image(mask.astype(np.float32), img.affine)
            nib.save(saved_image_mask, os.path.join(subject_output_dir, f'tensor_mask_{subject_id}.nii.gz'))
            
            if verbose:
                print(f"Successfully processed {subject_id}")


def process_sample_subject():
    """Process a sample subject for demonstration"""
    # Define paths for the sample subject
    base_dir = Path("sample_data")
    subject_id = "sample_subject"
    subject_dir = base_dir / subject_id
    output_dir = base_dir / "output"
    
    # Ensure directories exist
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample bvals/bvecs path
    bvals_path = subject_dir / f"{subject_id}_bval.npy"
    bvecs_path = subject_dir / f"{subject_id}_bvec.npy"
    nifti_path = subject_dir / f"4d_img_DTI_{subject_id}.nii.gz"
    
    if not nifti_path.exists():
        print("Sample data not found. Please download sample data first.")
        return False
    
    print(f"Processing sample subject: {subject_id}")
    
    # Create a temp directory structure similar to the full processing
    temp_input_dir = output_dir / "temp_input"
    temp_input_dir.mkdir(exist_ok=True)
    
    # Copy the sample NIfTI file to the temp input directory
    shutil.copy(str(nifti_path), str(temp_input_dir / nifti_path.name))
    
    # Process the sample subject
    try:
        # Use 46 as the default mode for sample
        process_subjects_by_mode(str(temp_input_dir), str(output_dir), 46, str(base_dir))
        print(f"Sample processing complete. Output saved to {output_dir}")
        return True
    except Exception as e:
        print(f"Error processing sample subject: {e}")
        return False


def main():
    """Main function to parse arguments and execute the script"""
    parser = argparse.ArgumentParser(description='Generate diffusion tensor parameters from diffusion MRI data')
    parser.add_argument('--subjects_path', type=str, help='Path to the subjects directory')
    parser.add_argument('--output_base', type=str, help='Path to the output base directory')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset containing reference bval/bvec files')
    parser.add_argument('--mode', type=int, choices=[46, 54, 55], help='Diffusion mode (46, 54, or 55)')
    parser.add_argument('--all_modes', action='store_true', help='Process all diffusion modes')
    parser.add_argument('--organize_only', action='store_true', help='Only organize files by dimension, no processing')
    parser.add_argument('--sample', action='store_true', help='Process sample subject')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.sample:
        process_sample_subject()
        return
    
    if not args.subjects_path or not args.output_base:
        parser.print_help()
        return
    
    # Ensure directories exist
    os.makedirs(args.output_base, exist_ok=True)
    
    # Organize files by dimension
    print("Organizing files by dimension...")
    dimension_files = copy_nifti_with_dimension_check(
        args.subjects_path, 
        args.output_base,
        verbose=args.verbose
    )
    
    if args.organize_only:
        print("Files organized successfully.")
        return
    
    if not args.dataset_path:
        print("Error: dataset_path is required for tensor parameter calculation")
        return
    
    # Process subjects
    if args.all_modes:
        modes = [46, 54, 55]
    elif args.mode:
        modes = [args.mode]
    else:
        print("Please specify either --mode or --all_modes")
        return
    
    for mode in modes:
        input_dir = os.path.join(args.output_base, f'all_{mode}_diff_4d_img_ordered')
        if os.path.exists(input_dir):
            process_subjects_by_mode(input_dir, args.output_base, mode, args.dataset_path, verbose=args.verbose)
        else:
            print(f"No files found for mode {mode}")


if __name__ == "__main__":
    main() 