import os

import numpy as np
from matplotlib import pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
from preprocess import fix_seed, read_and_preprocess_data
from metrics import metric
from train import Train
from utils import clustering
from params import params

def main(args):
    fix_seed(args.random_seed)
    data, adata_omics1 = read_and_preprocess_data(args)
    model = Train(data, args)
    output = model.train()
    adata = adata_omics1.copy()
    adata.obsm['SpaMSLA'] = output.copy()
    clustering(adata, key='SpaMSLA', add_key='SpaMSLA', n_clusters=args.n_clusters, method=args.tool, use_pca=True)
    metric(args, adata)

    if args.data_type == 'CITE':
        y_max = np.max(adata.obsm['spatial'][:, 1])
        adata.obsm['spatial'][:, 1] = y_max - adata.obsm['spatial'][:, 1]
    if args.data_type == 'H3K27ac' or args.data_type == 'H3K4me3':
        spatial_coords = adata.obsm['spatial'].copy()
        rotated_coords = np.column_stack([spatial_coords[:, 1], -spatial_coords[:, 0]])
        adata.obsm['spatial'] = rotated_coords
    if args.data_type == 'H3K27me3':
        spatial_coords = adata.obsm['spatial'].copy()
        rotated_coords = np.column_stack([spatial_coords[:, 1], -spatial_coords[:, 0]])
        rotated_coords[:, 1] = -rotated_coords[:, 1]
        adata.obsm['spatial'] = rotated_coords
    if 'MISAR' in args.data_type:
        adata.obsm['spatial'][:, 1] *= -1
    if 'SPOTS' in args.data_type:
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
    if args.data_type == 'Stereo_simu':
        spatial_coords = adata.obsm['spatial'].copy()
        rotated_coords = np.column_stack([-spatial_coords[:, 1], spatial_coords[:, 0]])
        adata.obsm['spatial'] = rotated_coords * np.array([-1, 1])
    import scanpy as sc
    sc.pl.embedding(adata, basis='spatial', color='SpaMSLA', s=100, show=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--data_type', type=str, default='STARmap', help='data_type')
    parser.add_argument('--random_seed', type=int, default=2024, help='random_seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--tool', type=str, default='mclust', choices=['mclust', 'leiden', 'louvain'], help='tool for clustering')
    parser.add_argument('--dim_output', type=int, default=64, help='dimension of output data')
    parser.add_argument('--pth', action='store_true', help='use trained weight')
    args = parser.parse_args()
    args = params(args)
    main(args)
