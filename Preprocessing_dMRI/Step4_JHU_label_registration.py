#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5: JHU Label Registration on Ordered Data

This script performs registration of JHU atlas labels onto subject-specific brain images.
It registers a template FA image to each subject's FA map, then applies the transformation
to JHU labels to get subject-specific labeled regions.

Usage:
    python Step5_JHU_label_registration.py --mode 46 --sample

Requirements:
    - Python packages: antspyx, dipy, nibabel, numpy
"""

import os
import shutil
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from dipy.io.image import load_nifti

# Try importing ants, handle potential import errors
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    print("ANTs library not available. Install with: pip install antspyx")
    ANTS_AVAILABLE = False


def register_single_subject(subject_id, fixed_image_path, output_label_path, warped_folder,
                          moving_image_path, label_image_path, verbose=True):
    """
    Register JHU labels to a single subject's FA image.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    fixed_image_path : str
        Path to the subject's FA image (target for registration)
    output_label_path : str
        Path where registered label image will be saved
    warped_folder : str
        Folder to save transformation files
    moving_image_path : str
        Path to the template FA image
    label_image_path : str
        Path to the JHU label image
    verbose : bool
        Whether to print detailed information
    
    Returns:
    --------
    bool
        Success status
    """
    if not ANTS_AVAILABLE:
        print("ANTs library required for registration")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    os.makedirs(warped_folder, exist_ok=True)
    
    try:
        # Load the fixed image (subject FA map)
        data, affine, img = load_nifti(fixed_image_path, return_img=True)
        if verbose:
            print(f"Processing subject: {subject_id}")
            print(f"Image shape: {data.shape}")
        
        # Load images for registration
        fixed = ants.image_read(fixed_image_path)
        moving = ants.image_read(moving_image_path)
        
        # Perform registration -> will generate warp
        mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
        
        # Get paths to the generated transforms
        warp_path = mytx['fwdtransforms'][0]
        affine_path = mytx['fwdtransforms'][1]
        
        # Apply transformations to the label image
        label_image = ants.image_read(label_image_path)
        transformed_label_image = ants.apply_transforms(
            fixed=fixed, 
            moving=label_image, 
            interpolator='nearestNeighbor', 
            transformlist=mytx['fwdtransforms']
        )
        
        # Save the transformed label image
        ants.image_write(transformed_label_image, output_label_path)
        
        # Save warp and affine transformation files
        warp_save_path = os.path.join(warped_folder, f"{subject_id}_warp.nii.gz")
        affine_save_path = os.path.join(warped_folder, f"{subject_id}_affine.mat")
        
        # Copy and rename warp and affine files
        ants.image_write(ants.image_read(warp_path), warp_save_path)
        with open(affine_path, 'rb') as affine_file:
            affine_data = affine_file.read()
        with open(affine_save_path, 'wb') as output_file:
            output_file.write(affine_data)
        
        if verbose:
            print(f"Saved warp file: {warp_save_path}")
            print(f"Saved affine matrix file: {affine_save_path}")
        
        return True
    
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return False


def register_subjects_by_mode(subjects_list, mode, input_base_path, output_base_path, 
                              moving_image_path, label_image_path, verbose=True):
    """
    Register a list of subjects with specific diffusion mode.
    
    Parameters:
    -----------
    subjects_list : list
        List of subject IDs to process
    mode : int
        Number of diffusion directions (46, 54, or 55)
    input_base_path : str
        Base path to input data
    output_base_path : str
        Base path for output data
    moving_image_path : str
        Path to the template FA image
    label_image_path : str
        Path to the JHU label image
    verbose : bool
        Whether to print detailed information
    """
    # Set up paths
    warped_folder = os.path.join(output_base_path, f"{mode}_diff_registered", "warped")
    os.makedirs(warped_folder, exist_ok=True)
    
    success_count = 0
    
    for subject_id in subjects_list:
        # Construct paths for current subject
        input_data_path = os.path.join(input_base_path, f"all_{mode}_diff_4d_img_ordered", f"4d_img_DTI_{subject_id}.nii.gz")
        fixed_image_path = os.path.join(input_base_path, f"DTI_parameters_{mode}_diff", subject_id, f"tensor_fa_{subject_id}.nii.gz")
        output_label_path = os.path.join(output_base_path, f"{mode}_diff_registered", f"registered_label_image_{subject_id}.nii.gz")
        
        # Check if files exist
        if not os.path.exists(input_data_path):
            print(f"Input data not found for subject {subject_id}: {input_data_path}")
            continue
            
        if not os.path.exists(fixed_image_path):
            print(f"Fixed image not found for subject {subject_id}: {fixed_image_path}")
            continue
        
        # Process subject
        try:
            if verbose:
                print(f"\nProcessing subject: {subject_id} (mode: {mode})")
                
            # Load 4D data to check dimensions
            fourd_data, _, _ = load_nifti(input_data_path, return_img=True)
            if verbose:
                print(f"4D data shape: {fourd_data.shape}")
            
            success = register_single_subject(
                subject_id=subject_id,
                fixed_image_path=fixed_image_path,
                output_label_path=output_label_path,
                warped_folder=warped_folder,
                moving_image_path=moving_image_path,
                label_image_path=label_image_path,
                verbose=verbose
            )
            
            if success:
                success_count += 1
                
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
    
    print(f"Successfully processed {success_count} of {len(subjects_list)} subjects for mode {mode}")


def process_sample_subject():
    """Process a sample subject for demonstration"""
    if not ANTS_AVAILABLE:
        print("ANTs library required for registration")
        return False
    
    # Define paths for the sample subject
    base_dir = Path("sample_data")
    subject_id = "sample_subject"
    
    subject_dir = base_dir / "output" / "DTI_parameters_46_diff" / subject_id
    output_dir = base_dir / "output" / "registered_data"
    warped_dir = output_dir / "warped"
    
    # Ensure directories exist
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    warped_dir.mkdir(parents=True, exist_ok=True)
    
    # Define necessary files
    fa_image_path = subject_dir / f"tensor_fa_{subject_id}.nii.gz"
    output_label_path = output_dir / f"registered_label_image_{subject_id}.nii.gz"
    
    # Define template files (these would normally be standard templates)
    moving_image_path = base_dir / "templates" / "FMRIB58_FA_1mm.nii.gz"
    label_image_path = base_dir / "templates" / "JHU-ICBM-labels-1mm.nii.gz"
    
    if not fa_image_path.exists() or not moving_image_path.exists() or not label_image_path.exists():
        print("Sample data not found. Please download sample data first.")
        return False
    
    print(f"Processing sample subject: {subject_id}")
    
    # Register the sample subject
    success = register_single_subject(
        subject_id=subject_id,
        fixed_image_path=str(fa_image_path),
        output_label_path=str(output_label_path),
        warped_folder=str(warped_dir),
        moving_image_path=str(moving_image_path),
        label_image_path=str(label_image_path)
    )
    
    if success:
        print(f"Sample processing complete. Output saved to {output_dir}")
        return True
    else:
        print("Sample processing failed.")
        return False


def main():
    """Main function to parse arguments and execute the script"""
    parser = argparse.ArgumentParser(description='Register JHU atlas labels to subject FA maps')
    parser.add_argument('--mode', type=int, choices=[46, 54, 55], help='Diffusion mode to process (46, 54, or 55)')
    parser.add_argument('--input_path', type=str, help='Path to the input base directory')
    parser.add_argument('--output_path', type=str, help='Path to the output base directory')
    parser.add_argument('--moving_image', type=str, help='Path to the template FA image (e.g., FMRIB58_FA_1mm.nii.gz)')
    parser.add_argument('--label_image', type=str, help='Path to the JHU label image (e.g., JHU-ICBM-labels-1mm.nii.gz)')
    parser.add_argument('--subjects_file', type=str, help='Path to a text file containing subject IDs')
    parser.add_argument('--subject', type=str, help='Process a single subject ID')
    parser.add_argument('--sample', action='store_true', help='Process sample subject')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.sample:
        process_sample_subject()
        return
    
    # Check for required dependencies
    if not ANTS_AVAILABLE:
        print("ANTs library is required but not available. Install with: pip install antspyx")
        return
    
    # Validate arguments
    if not args.mode:
        print("Please specify diffusion mode with --mode (46, 54, or 55)")
        parser.print_help()
        return
    
    if not args.input_path or not args.output_path or not args.moving_image or not args.label_image:
        print("Missing required paths. Please specify --input_path, --output_path, --moving_image, and --label_image")
        parser.print_help()
        return
    
    # Get list of subjects
    subjects_list = []
    
    if args.subject:
        # Process a single subject
        subjects_list = [args.subject]
    elif args.subjects_file:
        # Load subjects from file
        try:
            with open(args.subjects_file, 'r') as f:
                subjects_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading subjects file: {e}")
            return
    else:
        # Try to find subjects from directories
        mode = args.mode
        dti_params_dir = os.path.join(args.input_path, f"DTI_parameters_{mode}_diff")
        
        if os.path.exists(dti_params_dir):
            subjects_list = [d for d in os.listdir(dti_params_dir) if os.path.isdir(os.path.join(dti_params_dir, d))]
        
        if not subjects_list:
            print("No subjects found. Please specify --subjects_file or --subject")
            return
    
    print(f"Processing {len(subjects_list)} subjects for mode {args.mode}")
    
    # Process subjects
    register_subjects_by_mode(
        subjects_list=subjects_list,
        mode=args.mode,
        input_base_path=args.input_path,
        output_base_path=args.output_path,
        moving_image_path=args.moving_image,
        label_image_path=args.label_image,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main() 