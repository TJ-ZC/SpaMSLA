from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
import numpy as np
from sklearn import metrics
from s_dbw import S_Dbw

def metric(args, adata):
    if args.data_type in ['COSMOS-ATAC-RNA-seq', 'MISAR7', 'STARmap', 'Stereo_simu', 'HLN_A1']:
        GT = []
        with open(f'{args.file_fold}GT_labels.txt', 'r') as f:
            for line in f:
                num = int(line.strip())
                GT.append(num)
        GT_list = GT
        Our_list = adata.obs.SpaMSLA.tolist()
        print(min(GT_list), max(GT_list))
        print(min(Our_list), max(Our_list))
        print(set(GT_list))
        print(set(Our_list))
        print(len(GT_list))
        print(len(Our_list))
        print(f"MI: {mutual_info_score(GT_list, Our_list):.6f}")
        print(f"NMI: {normalized_mutual_info_score(GT_list, Our_list):.6f}")
        print(f"AMI: {adjusted_mutual_info_score(GT_list, Our_list):.6f}")
        print(f"V-measure: {v_measure_score(GT_list, Our_list):.6f}")
        print(f"Homogeneity: {homogeneity_score(GT_list, Our_list):.6f}")
        print(f"Completeness: {completeness_score(GT_list, Our_list):.6f}")
        print(f"ARI: {adjusted_rand_score(GT_list, Our_list):.6f}")
        print(f"FMI: {fowlkes_mallows_score(GT_list, Our_list):.6f}")
    elif args.data_type in ['CITE', 'H3K27ac', 'H3K27me3', 'H3K4me3', 'MISAR12', 'SPOTS1', 'SPOTS2']:
        X = adata.obsm['SpaMSLA_pca']
        y = adata.obs.SpaMSLA
        y = y.values.reshape(-1)
        y = y.codes
        CH = np.round(metrics.calinski_harabasz_score(X, y), 6)
        SH = np.round(metrics.silhouette_score(X, y), 6)
        sdbw = np.round(S_Dbw(X, y), 6)
        DB = np.round(metrics.davies_bouldin_score(X, y), 6)
        print('CH', CH)
        print('SH', SH)
        print('sdbw', sdbw)
        print('DB', DB)
