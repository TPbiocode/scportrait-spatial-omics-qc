#!/usr/bin/env python3

"""
scPortrait workflow from DAPI segmentation to single cell extraction and cell featurization for the full dataset

Run with:
    python scportrait_dapi_workflow.py
"""
###############################################################################
# Import Libraries
###############################################################################
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scportrait.pipeline.project import Project
from scportrait.pipeline.segmentation.workflows import DAPISegmentation # Standard Segmentation workflow for DAPI-stained (nuclear stained) images
from scportrait.pipeline.extraction import HDF5CellExtraction
from scportrait.pipeline.featurization import CellFeaturizer

import tifffile
import torch

import gc
import psutil
import sys

###############################################################################
# Memory Usage Function
###############################################################################
def print_rss(stage):
    gc.collect()
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    print(f"[RSS] {stage}: {rss:.2f} GB")

def main():

    ###############################################################################
    # Configuration
    ###############################################################################
    #SCRIPT_DIR = Path(__file__).resolve().parent
    #REPO_ROOT = SCRIPT_DIR.parent
    REPO_ROOT = Path("/scratch/tmurugan/scportrait-spatial-omics-qc")  


    PROJECT_DIR = REPO_ROOT / "data" / "interim" / "10x_xenium_breast_cancer_rep1_dapisegmentation_full_project"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")

    ###############################################################################
    # Load project
    ###############################################################################
    CONFIG_PATH = REPO_ROOT / "configs" / "dapi_segmentation_full_dataset_config.yml"

    project = Project(
        project_location=PROJECT_DIR,
        config_path=CONFIG_PATH,
        overwrite=True,
        debug=True,
        segmentation_f=DAPISegmentation,
        extraction_f=HDF5CellExtraction,
        featurization_f=CellFeaturizer
    )

    print("Project initialized")
    print_rss("After Project initialization")

    ###############################################################################
    # Load input tif file
    ###############################################################################

    tif_file_path = REPO_ROOT / "data" / "processed" / "breast-cancer-rep1-10x-xenium-bundle_full_dataset_morphology_focus_rescaled.ome.tif"

    img = tifffile.imread(tif_file_path)

    y, x = img.shape

    print(f"Image shape: {y} x {x} pixels")
    print(img.dtype)

    project.load_input_from_tif_files(file_paths = [tif_file_path], 
                                    channel_names=["DAPI"],
                                    crop=[(0, y), (0, x)],
                                    overwrite=True)

    print("Loaded input tif file into project")

    project.print_project_status()
    print(project.sdata)

    project.input_image
    #project.plot_input_image()

    print_rss("After loading TIFF")


    ###############################################################################
    # DAPI Segmentation
    ###############################################################################
    print("Input image shape:", project.input_image.shape)
    print("Input image dtype:", project.input_image.dtype)

    # Turn off debug mode for segmentation to avoid the segmentation pipeline from failing
    print(project.segmentation_f.debug)
    project.segmentation_f.debug = False

    # Segmentation
    project.segment()

    # SpatialData object after segmentation
    print(project.sdata)

    print_rss("After segmentation")

    # Compute the number of segmented nuclei
    from skimage.measure import regionprops

    nuc = project.sdata.labels["seg_all_nucleus"]["scale0"]["image"].data.compute()

    print("Number of segmented nuclei:", len(regionprops(nuc)))

    # Compute the segmented nuclei area statistics
    areas = [r.area for r in regionprops(nuc)]

    print("Count:", len(areas))
    print("Median:", np.median(areas))
    print("5th percentile:", np.percentile(areas, 5))
    print("95th percentile:", np.percentile(areas, 95))

    #project.plot_segmentation_masks()

    print_rss("After regionprops")

    print("DAPI Segmentation completed")

    del nuc
    del areas

    gc.collect()

    ###############################################################################
    # Single-Cell Extraction
    ###############################################################################
    print_rss("Before extraction and after cleanup")

    # Memory profiling before extraction step using pympler
    from pympler import muppy, summary

    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)

    # Memory profiling before extraction step using guppy3
    from guppy import hpy

    h = hpy()
    print(h.heap())

    # Extract single cell images from the segmented nuclei
    project.extract()

    # Show the created AnnData object after single cell image extraction
    print(project.h5sc)

    # Visualize the extracted single cell images for randomly selected cell IDs
    #project.plot_single_cell_images()

    print_rss("After extraction")

    ###############################################################################
    # Featurization using CellFeaturizer
    ###############################################################################
    print_rss("Before featurization")

    project.featurize(overwrite = True)

    # Examine the SpatialData object after featurization
    print(project.sdata)

    # View the extracted features for the segmented nuclei
    results = project.sdata["CellFeaturizer_nucleus"].to_df()
    print(results.head())

    print_rss("After featurization")
    print("Cell featurization completed.")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()