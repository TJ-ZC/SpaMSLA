import os

import h5py
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

'''
---------------------
author: Yahui Long https://github.com/JinmiaoChenLab/SpatialGlue
e-mail: chen_jinmiao@bii.a-star.edu.sg
AGPL-3.0 LICENSE
---------------------
'''

def construct_neighbor_graph(adata_omics1, adata_omics2, datatype=None, n_neighbors=10):

    """
    Construct spatial graph. 

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['H3K27ac', 'H3K27me3', 'H3K4me3', 'COSMOS-ATAC-RNA-seq', 'MISAR7', 'MISAR12']:
        n_neighbors = 6
    elif datatype in ['CITE', 'STARmap', 'Stereo_simu']:
        n_neighbors = 10
    elif datatype in ['HLN_A1', 'SPOTS1', 'SPOTS2']:
        n_neighbors = 3
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data


def pca(adata, use_reps=None, n_comps=10):

    """Dimension reduction with PCA algorithm"""

    pca = PCA(n_components=n_comps)

    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def clr_normalize_each_cell(adata, inplace=True):

    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def construct_graph_by_coordinate(cell_position, n_neighbors=3):

    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    adj_spatial_omics1 = adj_spatial_omics1.toarray()  # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()

    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2 > 1, 1, adj_spatial_omics2)

    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1)  # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)

    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           }

    return adj


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def fix_seed(seed):
    # seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def read_and_preprocess_data(args):
    adata_omics1, adata_omics2 = None, None
    if args.data_type == 'HLN_A1' or 'SPOTS' in args.data_type:
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_ADT.h5ad')
    elif 'H3K' in args.data_type:
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_peaks_normalized.h5ad')
    elif args.data_type == 'COSMOS-ATAC-RNA-seq':
        data_mat = h5py.File(args.file_fold + 'ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5', 'r')
        df_data_RNA = np.array(data_mat['X_RNA']).astype('float64')
        df_data_ATAC = np.array(data_mat['X_ATAC']).astype('float64')
        loc = np.array(data_mat['Pos']).astype('float64')
        adata_omics1 = sc.AnnData(df_data_RNA, dtype="float64")
        adata_omics1.obsm['spatial'] = np.array(loc)
        adata_omics2 = sc.AnnData(df_data_ATAC, dtype="float64")
        adata_omics2.obsm['spatial'] = np.array(loc)
    elif 'MISAR' in args.data_type:
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_ATAC.h5ad')
    elif args.data_type == 'STARmap' or args.data_type == 'Stereo_simu':
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_simu1.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_simu2.h5ad')
    elif args.data_type == 'CITE':
        data_mat = h5py.File(args.file_fold + 'Spatial_CITE_seq_HumanTonsil_RNA_Protein.h5')
        df_data_RNA = np.array(data_mat['X_gene']).astype('float64')
        df_data_protein = np.array(data_mat['X_protein']).astype('float64')
        loc = np.array(data_mat['pos']).astype('float64')
        gene_names = list(data_mat['gene'])
        gene_names = [gene.decode("utf-8") for gene in gene_names]
        protein_names = list(data_mat['protein'])
        protein_names = [protein.decode("utf-8") for protein in protein_names]
        protein_names = [protein.split(".")[0] for protein in protein_names]
        adata_omics1 = sc.AnnData(df_data_RNA, dtype="float64")
        adata_omics1.index = gene_names
        adata_omics2 = sc.AnnData(df_data_protein, dtype="float64")
        adata_omics2.index = protein_names
        adata_omics1.obsm['spatial'] = np.array(loc)
        adata_omics2.obsm['spatial'] = np.array(loc)

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    fix_seed(args.random_seed)

    # Preprocess
    if args.data_type == 'HLN_A1':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)
    elif 'H3K' in args.data_type:
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=200)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        # ATAC
        adata_omics2 = adata_omics2[
            adata_omics1.obs_names].copy()  # .obsm['X_lsi'] represents the dimension reduced feature
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)
    elif 'SPOTS' in args.data_type:
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)
    elif args.data_type == 'COSMOS-ATAC-RNA-seq':
        adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=50)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)
    elif 'MISAR' in args.data_type:
        sc.pp.filter_genes(adata_omics1, min_cells=50)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=20)
        sc.pp.filter_genes(adata_omics2, min_cells=50)
        sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics2, target_sum=1e4)
        sc.pp.log1p(adata_omics2)
        adata_omics2_high = adata_omics2[:, adata_omics2.var['highly_variable']]
        adata_omics2.obsm['feat'] = pca(adata_omics2_high, n_comps=20)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)
    elif args.data_type == 'STARmap' or args.data_type == 'Stereo_simu':
        adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=50)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)
    elif args.data_type == 'CITE':
        sc.pp.normalize_per_cell(adata_omics1)
        sc.pp.log1p(adata_omics1)
        sc.pp.log1p(adata_omics2)
        adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=50)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type)
    else:
        assert 0

    return data, adata_omics1
