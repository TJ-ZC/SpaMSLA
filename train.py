import torch
from torch_geometric.typing import SparseTensor
import torch.nn.functional as F
from tqdm import tqdm
from model import SpaMSLA
from preprocess import adjacent_matrix_preprocessing


class Train:
    def __init__(self,
                 data,
                 args=None
                 ):

        self.data = data.copy()
        self.datatype = args.data_type
        self.device = args.device
        self.random_seed = args.random_seed
        self.dim_output = args.dim_output

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to_dense().to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to_dense().to(self.device)

        self.spatial_edge_list = self.adj['adj_spatial_omics1']._indices().to(self.device)
        self.spatial_edge_weight = self.adj['adj_spatial_omics1']._values().to(self.device)

        self.arg = args

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        self.epochs = self.arg.epochs

        self.spatial_sp = SparseTensor(row=self.spatial_edge_list[0], col=self.spatial_edge_list[1], value=self.spatial_edge_weight)

    def train(self):

        self.model = SpaMSLA(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2, self.arg.hidden_feat, self.arg.dropout).to(self.device)
        self.optimizer = torch.optim.SGD(list(self.model.parameters()),
                                         lr=self.arg.learning_rate,
                                         momentum=0.9,
                                         weight_decay=self.arg.weight_decay)

        self.model.train()
        if self.arg.pth:
            self.model = torch.load(f'./pth/{self.arg.data_type}.pth')
        else:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_spatial_omics2, self.spatial_sp)
                self.rec_feature1 = F.mse_loss(self.features_omics1, results['decoder_out1'])
                self.rec_feature2 = F.mse_loss(self.features_omics2, results['decoder_out2'])
                loss = self.rec_feature1 * self.arg.weight1 + self.rec_feature2 * self.arg.weight2 + results['ctr'] * self.arg.weight3
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_spatial_omics2, self.spatial_sp)

        emb_combined = F.normalize(results['attn_omics'], p=2, eps=1e-12, dim=1)

        return emb_combined.detach().cpu().numpy()
