#!/usr/bin/env python3

"""
scPortrait workflow using extracted single cell images for getting ConvNeXT embeddings for the full dataset

Run with:
    python scportrait_dapi_convnext.py
"""
###############################################################################
# Import Libraries
###############################################################################
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scportrait.pipeline.project import Project
from scportrait.pipeline.extraction import HDF5CellExtraction
from scportrait.pipeline.featurization import ConvNeXtFeaturizer

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
        extraction_f=HDF5CellExtraction,
        featurization_f=ConvNeXtFeaturizer
    )

    project.print_project_status()
    print(project.extraction_f)
    print(project.featurization_f)

    print("Project initialized")
    print_rss("After Project initialization")

    ###############################################################################
    # Featurization using ConvNeXtFeaturizer
    ###############################################################################
    print_rss("Before featurization")

    project.featurize(overwrite = True)

    # Examine the SpatialData object after featurization
    print(project.sdata)

    # View the extracted features for the segmented nuclei
    results = project.sdata["ConvNeXtFeaturizer_Ch1_nucleus"].to_df()
    print(results.head())

    print_rss("After featurization")
    print("ConvNeXt featurization completed.")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()