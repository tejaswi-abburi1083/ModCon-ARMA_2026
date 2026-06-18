#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 1: Converting DICOM to NIfTI format for diffusion MRI data

This script converts raw DICOM files to the NIfTI format required for further processing.
It handles the conversion of diffusion MRI data and associated gradient files (bval/bvec).

Usage:
    python Step1_DICOM_to_NIfTI_conversion.py --input_dir /path/to/raw/data --output_dir /path/to/output

Requirements:
    - dcm2niix: A DICOM to NIfTI converter (https://github.com/rordenlab/dcm2niix)
    - Python packages: os, numpy, pandas, subprocess, nibabel
"""

import os
import pandas as pd
import subprocess
import numpy as np
import shutil
import argparse
import nibabel as nib
from pathlib import Path


def get_subject_details(subject_id, metadata_file=None):
    """
    Extract subject details from metadata file or filename.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., '016_S_6839')
    metadata_file : str, optional
        Path to CSV file with subject metadata
        
    Returns:
    --------
    dict
        Dictionary containing age, sex, and group information
    """
    # If metadata file is provided, extract details from it
    if metadata_file and os.path.exists(metadata_file):
        try:
            df = pd.read_csv(metadata_file)
            subject_info = df[df['Subject'] == subject_id]
            if not subject_info.empty:
                return {
                    'age': subject_info['Age'].values[0],
                    'sex': subject_info['Sex'].values[0],
                    'group': subject_info['Group'].values[0],
                    'modality': 'DTI'
                }
        except Exception as e:
            print(f"Error reading metadata file: {e}")
    
    # Default values if metadata file is not available or subject not found
    # Extract group from directory name or use default
    if '_AD_' in subject_id or subject_id.endswith('_AD'):
        group = 'AD'
    elif '_MCI_' in subject_id or subject_id.endswith('_MCI'):
        group = 'MCI'
    else:
        group = 'CN'  # Default to Cognitive Normal
    
    # Try to extract sex and age from filename if in format: XXX_S_XXXX_[M/F]_XX
    parts = subject_id.split('_')
    sex = 'U'  # Unknown by default
    age = 0    # Unknown by default
    
    for i, part in enumerate(parts):
        if part in ['M', 'F'] and i < len(parts) - 1 and parts[i+1].isdigit():
            sex = part
            age = int(parts[i+1])
            break
    
    return {
        'age': age,
        'sex': sex,
        'group': group,
        'modality': 'DTI'
    }


def is_already_processed(subject_dir, output_dir):
    """
    Check if a subject has already been processed.
    
    Parameters:
    -----------
    subject_dir : str
        Path to the subject's input directory
    output_dir : str
        Path to the subject's output directory
        
    Returns:
    --------
    bool
        True if already processed, False otherwise
    """
    subject_id = os.path.basename(subject_dir)
    
    # Check if the output directory exists and contains required files
    if not os.path.exists(output_dir):
        return False
    
    # Look for NIfTI, bval, and bvec files
    nifti_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz') and subject_id in f]
    bval_files = [f for f in os.listdir(output_dir) if f.endswith('.bval') and subject_id in f]
    bvec_files = [f for f in os.listdir(output_dir) if f.endswith('.bvec') and subject_id in f]
    
    return len(nifti_files) > 0 and len(bval_files) > 0 and len(bvec_files) > 0


def convert_dmri_data(subject_dir, output_dir, metadata_file=None):
    """
    Convert DICOM files to NIfTI format for a subject.
    
    Parameters:
    -----------
    subject_dir : str
        Path to the subject's input directory
    output_dir : str
        Path to the output directory
    metadata_file : str, optional
        Path to CSV file with subject metadata
        
    Returns:
    --------
    bool
        True if conversion successful, False otherwise
    """
    subject_id = os.path.basename(subject_dir)
    print(f"Converting subject: {subject_id}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for conversion
    temp_dir = os.path.join(output_dir, 'temp_conversion')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Get subject details
        details = get_subject_details(subject_id, metadata_file)
        
        # Find DICOM files - look for Axial_DTI folder or similar
        dicom_dir = None
        for root, dirs, files in os.walk(subject_dir):
            potential_dirs = [d for d in dirs if 'DTI' in d or 'dti' in d or 'dwi' in d or 'DWI' in d]
            if potential_dirs:
                dicom_dir = os.path.join(root, potential_dirs[0])
                break
        
        if not dicom_dir:
            # If no specialized folder found, use the subject directory
            dicom_dir = subject_dir
            
            # Check if there are any DICOM files in the directory
            dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            if not dicom_files:
                print(f"No DICOM files found for subject {subject_id}")
                return False
        
        # Run dcm2niix conversion
        cmd = [
            'dcm2niix',
            '-f', f"{subject_id}_%p",  # Output filename format
            '-z', 'y',                 # Compress output (.nii.gz)
            '-b', 'y',                 # Generate bval/bvec files
            '-o', temp_dir,            # Output directory
            dicom_dir                  # Input directory
        ]
        
        subprocess.run(cmd, check=True)
        
        # Check for output files
        nifti_files = [f for f in os.listdir(temp_dir) if f.endswith('.nii.gz')]
        bval_files = [f for f in os.listdir(temp_dir) if f.endswith('.bval')]
        bvec_files = [f for f in os.listdir(temp_dir) if f.endswith('.bvec')]
        
        if not (nifti_files and bval_files and bvec_files):
            print(f"Conversion failed for {subject_id} - missing output files")
            return False
        
        # Create standardized filenames
        final_nifti = f"{subject_id}_{details['group']}_{details['sex']}_{details['age']}_4d.nii.gz"
        final_bval = f"{subject_id}_{details['group']}_{details['sex']}_{details['age']}.bval"
        final_bvec = f"{subject_id}_{details['group']}_{details['sex']}_{details['age']}.bvec"
        
        # Move and rename files
        shutil.move(os.path.join(temp_dir, nifti_files[0]), os.path.join(output_dir, final_nifti))
        shutil.move(os.path.join(temp_dir, bval_files[0]), os.path.join(output_dir, final_bval))
        shutil.move(os.path.join(temp_dir, bvec_files[0]), os.path.join(output_dir, final_bvec))
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        print(f"Successfully converted subject {subject_id}")
        return True
        
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def verify_conversion(output_dir, subject_id):
    """
    Verify that the conversion produced valid NIfTI, bval, and bvec files.
    
    Parameters:
    -----------
    output_dir : str
        Path to the output directory
    subject_id : str
        Subject identifier
        
    Returns:
    --------
    bool
        True if verification passed, False otherwise
    """
    print(f"Verifying conversion for subject {subject_id}")
    
    # Find files for this subject
    nifti_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz') and subject_id in f]
    bval_files = [f for f in os.listdir(output_dir) if f.endswith('.bval') and subject_id in f]
    bvec_files = [f for f in os.listdir(output_dir) if f.endswith('.bvec') and subject_id in f]
    
    if not (nifti_files and bval_files and bvec_files):
        print("Missing files:")
        if not nifti_files: print("  - NIfTI file")
        if not bval_files: print("  - bval file")
        if not bvec_files: print("  - bvec file")
        return False
    
    try:
        # Check NIfTI file
        nifti_path = os.path.join(output_dir, nifti_files[0])
        img = nib.load(nifti_path)
        shape = img.shape
        
        # Check bval/bvec files
        bval_path = os.path.join(output_dir, bval_files[0])
        bvec_path = os.path.join(output_dir, bvec_files[0])
        bvals = np.loadtxt(bval_path)
        bvecs = np.loadtxt(bvec_path)
        
        # Verify dimensions match
        if len(shape) != 4:
            print(f"Warning: NIfTI file is not 4D (found {len(shape)}D)")
            return False
        
        directions = shape[3]
        if directions != len(bvals):
            print(f"Warning: Number of directions in NIfTI ({directions}) does not match bvals ({len(bvals)})")
            return False
        
        if directions != bvecs.shape[1]:
            print(f"Warning: Number of directions in NIfTI ({directions}) does not match bvecs ({bvecs.shape[1]})")
            return False
        
        print(f"Verification passed:")
        print(f"  - NIfTI shape: {shape}")
        print(f"  - Number of b-values: {len(bvals)}")
        print(f"  - bvecs shape: {bvecs.shape}")
        return True
        
    except Exception as e:
        print(f"Error verifying conversion: {e}")
        return False


def process_dataset(input_dir, output_dir, metadata_file=None):
    """
    Process an entire dataset, converting DICOM to NIfTI for each subject.
    
    Parameters:
    -----------
    input_dir : str
        Path to the input directory containing subject folders
    output_dir : str
        Path to the output directory
    metadata_file : str, optional
        Path to CSV file with subject metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of subject directories
    subjects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not subjects:
        print(f"No subject directories found in {input_dir}")
        return
    
    print(f"Found {len(subjects)} subjects in the dataset")
    
    # Process each subject
    successful = 0
    for subject_id in subjects:
        subject_dir = os.path.join(input_dir, subject_id)
        subject_output_dir = os.path.join(output_dir, subject_id)
        
        # Skip if already processed
        if is_already_processed(subject_dir, subject_output_dir):
            print(f"Skipping subject {subject_id}: Already processed")
            successful += 1
            continue
        
        # Convert DICOM to NIfTI
        if convert_dmri_data(subject_dir, subject_output_dir, metadata_file):
            # Verify conversion
            if verify_conversion(subject_output_dir, subject_id):
                successful += 1
    
    print(f"Processing complete: {successful}/{len(subjects)} subjects processed successfully")


def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(description='Convert DICOM to NIfTI for diffusion MRI data')
    parser.add_argument('--input_dir', required=True, help='Path to input directory containing subject folders')
    parser.add_argument('--output_dir', required=True, help='Path to output directory')
    parser.add_argument('--metadata', help='Path to CSV file with subject metadata')
    parser.add_argument('--subject', help='Process specific subject (optional)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Process single subject or entire dataset
    if args.subject:
        subject_dir = os.path.join(args.input_dir, args.subject)
        subject_output_dir = os.path.join(args.output_dir, args.subject)
        
        if not os.path.exists(subject_dir):
            print(f"Error: Subject directory {subject_dir} does not exist")
            return
        
        if is_already_processed(subject_dir, subject_output_dir):
            print(f"Subject {args.subject} already processed")
        else:
            if convert_dmri_data(subject_dir, subject_output_dir, args.metadata):
                verify_conversion(subject_output_dir, args.subject)
    else:
        process_dataset(args.input_dir, args.output_dir, args.metadata)


if __name__ == "__main__":
    # Example usage:
    # python Step1_DICOM_to_NIfTI_conversion.py --input_dir ../../Dataset/raw --output_dir ../../Dataset/processed
    main() 