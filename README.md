## Requirement
### Conda envs
```javascript
conda create -n SpaMSLA python=3.8.19
conda activate SpaMSLA
conda install pytorch==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install r-base=4.4.1
pip install torch-geometric==2.6.1
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install scanpy==1.9.1
pip install pandas==1.5.0
pip install numpy==1.22.3
conda install matplotlib=3.4.3
pip install --user scikit-misc
pip install leidenalg
pip install s-dbw
pip install rpy2==3.4.1
```
### R envs
```javascript
install.packages("mclust")
```
## Reproduce
We recommend using the training weights or embeddings we provide. Generally speaking, using embeddings for clustering can yield consistent results.
```javascript
python main.py --data_type STARmap --pth
```
or
```javascript
from utils import clustering
import numpy as np
import anndata as ad
emb = np.load('./embs/STARmap_pca.npy')
adata = ad.AnnData(emb)
adata.obsm['SpaMSLA_pca'] = emb
clustering(adata, n_clusters=6, key='SpaMSLA_pca', add_key='SpaMSLA', method='leiden')
GT = []
with open('./Data/simu_STARmap/GT_labels.txt', 'r') as f:
    for line in f:
        num = int(line.strip())
        GT.append(num)

GT_list = GT
Our_list = adata.obs.SpaMSLA.tolist()
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(GT_list, Our_list)
```
