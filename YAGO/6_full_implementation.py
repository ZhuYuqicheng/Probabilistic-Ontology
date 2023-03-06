# %% package
import pandas as pd
import torch
from torch import nn
from torch.distributions import uniform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import os

#%% Data processing
class NF_dataset(Dataset):
    def __init__(self, mode: str, filter_option: dict) -> None:
        self.fit_label_encoder()
        self.fit_relation_encoder()
        self.filter_option = filter_option
        if mode == "all":
            NF1_X, NF1_y = self.load_NF1()
            NF2_X, NF2_y = self.load_NF2()
            NF3_X, NF3_y = self.load_NF3()
            NF4_X, NF4_y = self.load_NF4()
            self.X = torch.cat((NF1_X, NF2_X, NF3_X, NF4_X), dim=0)
            self.y = torch.cat((NF1_y, NF2_y, NF3_y, NF4_y), dim=0)
        elif mode == "NF1":
            NF1_X, NF1_y = self.load_NF1()
            self.X = NF1_X
            self.y = NF1_y
        elif mode == "NF2":
            NF2_X, NF2_y = self.load_NF2()
            self.X = NF2_X
            self.y = NF2_y
        elif mode == "NF3":
            NF3_X, NF3_y = self.load_NF3()
            self.X = NF3_X
            self.y = NF3_y
        elif mode == "NF4":
            NF4_X, NF4_y = self.load_NF4()
            self.X = NF4_X
            self.y = NF4_y
        else:
            raise Exception("Please specify import mode!")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
    
    # ----------------------------------------------------------#
    
    def fit_label_encoder(self):
        class_table = pd.read_csv("count_A.csv")
        self.cls_le = preprocessing.LabelEncoder()
        self.cls_le.fit(class_table["class"].unique())

    def fit_relation_encoder(self):
        facts = pd.read_csv("wordnet_person_facts.csv")
        self.rel_le = preprocessing.LabelEncoder()
        self.rel_le.fit(facts["predicate"].unique())

    def NF_data_filter(self, NF_data):
        # drop rows containing nan values
        NF_data.dropna(inplace=True)
        NF_data = NF_data[NF_data["Probability"]!=0] if self.filter_option["filter_zero"] else NF_data
        NF_data = NF_data[NF_data["Probability"]>10e-3] if self.filter_option["filter_small"] else NF_data
        NF_data = NF_data[NF_data["Probability"]!=1] if self.filter_option["filter_one"] else NF_data

        NF_data.reset_index(inplace=True)
        return NF_data

    def load_NF1(self):
        NF1_data = pd.read_csv("./YAGO3_result/result_NF1.csv")
        # data processing
        NF1_data = self.NF_data_filter(NF1_data)
        # deal with X
        X_from_data = NF1_data[["ConceptA", "ConceptB"]].apply(self.cls_le.transform).values
        empty_vector = np.zeros((X_from_data.shape[0],1))
        ID = np.ones((X_from_data.shape[0],1))
        NF1_X = torch.tensor(np.concatenate((ID, empty_vector, X_from_data), axis=1))
        # deal with y
        NF1_y = torch.tensor(NF1_data["Probability"]).reshape((-1,1))
        return NF1_X, NF1_y
    
    def load_NF2(self):
        NF2_data = pd.read_csv("./YAGO3_result/result_NF2.csv")
        # data processing
        NF2_data = self.NF_data_filter(NF2_data)
        # deal with X
        X_from_data = NF2_data[["ConceptA", "ConceptB", "ConceptC"]].apply(self.cls_le.transform).values
        ID = np.ones((X_from_data.shape[0],1))*2
        NF2_X = torch.tensor(np.concatenate((ID, X_from_data), axis=1))
        # deal with y
        NF2_y = torch.tensor(NF2_data["Probability"]).reshape((-1,1))
        return NF2_X, NF2_y
    
    def load_NF3(self):
        NF3_data = pd.read_csv("./YAGO3_result/result_NF3.csv")
        # data processing
        NF3_data = self.NF_data_filter(NF3_data)
        # deal with X
        class_data = NF3_data[["ConceptA", "ConceptB"]].apply(self.cls_le.transform).values
        relation_data = NF3_data[["relation"]].apply(self.rel_le.transform).values
        ID = np.ones((class_data.shape[0],1))*3
        NF3_X = torch.tensor(np.concatenate((ID, class_data, relation_data), axis=1))
        # deal with y
        NF3_y = torch.tensor(NF3_data["Probability"]).reshape((-1,1))
        return NF3_X, NF3_y
    
    def load_NF4(self):
        NF4_data = pd.read_csv("./YAGO3_result/result_NF4.csv")
        # data processing
        NF4_data = self.NF_data_filter(NF4_data)
        # deal with X
        class_data = NF4_data[["ConceptA", "ConceptB"]].apply(self.cls_le.transform).values
        relation_data = NF4_data[["relation"]].apply(self.rel_le.transform).values
        ID = np.ones((class_data.shape[0],1))*4
        NF4_X = torch.tensor(np.concatenate((ID, class_data, relation_data), axis=1))
        # deal with y
        NF4_y = torch.tensor(NF4_data["Probability"]).reshape((-1,1))
        return NF4_X, NF4_y

# %% class
class Box:
    """
    A class used to represent a Box Embedding for class in ontology.

    Axis-parallel boxes are used to embed ontology classes, which are defined
    by two vectors, i.e., lower left corner and upper right corner.

    Attributes
    -----
    min_embed:
        the representation of lower left corner of the box
    max_embed:
        the representation of upper right corner of the box
    """
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed

class StatBoxEL(nn.Module):
    def __init__(self, size_dict, embed_dim, init_dict, volume_metric) -> None:
        super(StatBoxEL, self).__init__()
        min_embeddings = self.init_embedding(size_dict["vocab"], embed_dim, init_dict["min_embed"])
        max_embeddings = self.init_embedding(size_dict["vocab"], embed_dim, init_dict["max_embed"])
        rel_scale_embeddings = self.init_embedding(size_dict["relation"], embed_dim, init_dict["rel_scale"])
        rel_trans_embeddings = self.init_embedding(size_dict["relation"], embed_dim, init_dict["rel_trans"])

        self.min_embeddings = nn.Parameter(min_embeddings)
        self.max_embeddings = nn.Parameter(max_embeddings)
        self.rel_scale_embeddings = nn.Parameter(rel_scale_embeddings)
        self.rel_trans_embeddings = nn.Parameter(rel_trans_embeddings)

        self.volume_metric = volume_metric

    def forward(self, x):
        # check the tensor
        NF1_data = x[x[:,0] == 1]
        NF2_data = x[x[:,0] == 2]
        NF3_data = x[x[:,0] == 3]
        NF4_data = x[x[:,0] == 4]

        pred_list = []
        if NF1_data.numel(): pred_list.append(self.NF1_pred(NF1_data))
        if NF2_data.numel(): pred_list.append(self.NF2_pred(NF2_data))
        if NF3_data.numel(): pred_list.append(self.NF3_pred(NF3_data))
        if NF4_data.numel(): pred_list.append(self.NF4_pred(NF4_data))

        return torch.cat(pred_list, 0)

    def init_embedding(self, vocab_size:int, embed_dim:int, init_value:list):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        embedding = distribution.sample((vocab_size, embed_dim))
        return embedding
    
    def volumes(self, box):
        if self.volume_metric == "multi":
            return box.delta_embed.prod(1, keepdim=True)
        elif self.volume_metric == "manhattan":
            return box.delta_embed.sum(1, keepdim=True)
        elif self.volume_metric == "square_sum":
            return torch.square(box.delta_embed).sum(1, keepdim=True)
        elif self.volume_metric == "euclidean":
            return torch.sqrt(torch.square(box.delta_embed).sum(1, keepdim=True))
        
    def intersection(self, class_A, class_B):
        min_embed = torch.max(class_A.min_embed, class_B.min_embed)
        max_embed = torch.min(class_A.max_embed, class_B.max_embed)
        intersected_box = Box(min_embed, max_embed)
        return intersected_box
    
    def relation_mapping(self, relation_index, class_box):
        rel_scale = torch.index_select(self.rel_scale_embeddings, 0, relation_index)
        rel_trans = torch.index_select(self.rel_trans_embeddings, 0, relation_index)

        min_embed = class_box.min_embed * rel_scale + rel_trans
        max_embed = class_box.max_embed * rel_scale + rel_trans
        mapped_box = Box(min_embed, max_embed)
        return mapped_box

    def NF1_pred(self, NF1_data):
        class_index_A = NF1_data[:,2].to(torch.int)
        NF1_min_A = torch.index_select(self.min_embeddings, 0, class_index_A)
        NF1_max_A = torch.index_select(self.max_embeddings, 0, class_index_A)
        NF1_class_A = Box(NF1_min_A, NF1_max_A)

        class_index_B = NF1_data[:,3].to(torch.int)
        NF1_min_B = torch.index_select(self.min_embeddings, 0, class_index_B)
        NF1_max_B = torch.index_select(self.max_embeddings, 0, class_index_B)
        NF1_class_B = Box(NF1_min_B, NF1_max_B)

        class_AB = self.intersection(NF1_class_A, NF1_class_B)
        pred_p = self.volumes(class_AB)/self.volumes(NF1_class_A)
        return pred_p
    
    def NF2_pred(self, NF2_data):
        class_index_A1 = NF2_data[:,1].to(torch.int)
        NF2_min_A1 = torch.index_select(self.min_embeddings, 0, class_index_A1)
        NF2_max_A1 = torch.index_select(self.max_embeddings, 0, class_index_A1)
        class_A1 = Box(NF2_min_A1, NF2_max_A1)

        class_index_A2 = NF2_data[:,2].to(torch.int)
        NF2_min_A2 = torch.index_select(self.min_embeddings, 0, class_index_A2)
        NF2_max_A2 = torch.index_select(self.max_embeddings, 0, class_index_A2)
        class_A2 = Box(NF2_min_A2, NF2_max_A2)

        class_index_B = NF2_data[:,3].to(torch.int)
        NF2_min_B = torch.index_select(self.min_embeddings, 0, class_index_B)
        NF2_max_B = torch.index_select(self.max_embeddings, 0, class_index_B)
        class_B = Box(NF2_min_B, NF2_max_B)

        class_A1A2 = self.intersection(class_A1, class_A2)
        class_A1A2B = self.intersection(class_A1A2, class_B)
        pred_p = self.volumes(class_A1A2B)/self.volumes(class_A1A2)
        return pred_p
    
    def NF3_pred(self, NF3_data):
        class_index_A = NF3_data[:,1].to(torch.int)
        NF3_min_A = torch.index_select(self.min_embeddings, 0, class_index_A)
        NF3_max_A = torch.index_select(self.max_embeddings, 0, class_index_A)
        class_A = Box(NF3_min_A, NF3_max_A)
        
        class_index_B = NF3_data[:,2].to(torch.int)
        NF3_min_B = torch.index_select(self.min_embeddings, 0, class_index_B)
        NF3_max_B = torch.index_select(self.max_embeddings, 0, class_index_B)
        class_B = Box(NF3_min_B, NF3_max_B)

        relation_index = NF3_data[:,3].to(torch.int)
        class_rB = self.relation_mapping(relation_index, class_B)

        class_ArB = self.intersection(class_A, class_rB)
        pred_p = self.volumes(class_ArB)/self.volumes(class_A)
        return pred_p

    def NF4_pred(self, NF4_data):
        class_index_A = NF4_data[:,1].to(torch.int)
        NF3_min_A = torch.index_select(self.min_embeddings, 0, class_index_A)
        NF3_max_A = torch.index_select(self.max_embeddings, 0, class_index_A)
        class_A = Box(NF3_min_A, NF3_max_A)
        
        class_index_B = NF4_data[:,2].to(torch.int)
        NF3_min_B = torch.index_select(self.min_embeddings, 0, class_index_B)
        NF3_max_B = torch.index_select(self.max_embeddings, 0, class_index_B)
        class_B = Box(NF3_min_B, NF3_max_B)

        relation_index = NF4_data[:,3].to(torch.int)
        class_rB = self.relation_mapping(relation_index, class_B)

        class_ArB = self.intersection(class_A, class_rB)
        pred_p = self.volumes(class_ArB)/self.volumes(class_rB)
        return pred_p
    
#%% loss function
class MeanAbsoluteError(nn.Module):
    def __init__(self):
        super(MeanAbsoluteError, self).__init__()

    def forward(self, output, target):
        criterion = torch.nn.L1Loss()
        loss = criterion(output, target)
        return loss
    
def box_regularization(model, C):
    """
    make sure that min_embed < max_embed
    """
    param_dict = dict()
    for name, param in model.named_parameters():
        param_dict[name] = param
    delta_embedding = param_dict["max_embeddings"] - param_dict["min_embeddings"]
    minus_loss = abs(delta_embedding[delta_embedding<=0].sum().item())
    return minus_loss*C

#%% Training Process
def train(dataloader, model, criterion, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    acc_loss = 0
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Forward propagation
        pred = model(X)
        # Loss calculation (and regularization)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # monitoring
        running_loss += loss.item()
        # calculate accumulate training loss
        acc_loss += loss.item()
        if batch % 100 == 99:
            loss_num, current = running_loss/100, (batch + 1) * len(X)
            running_loss = 0
            print(f"Batch Loss: {loss_num:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Training Loss: {acc_loss/(batch + 1)}")
    return acc_loss/(batch + 1)

def test(mode, model, criterion):
    filter_option = {"filter_zero": False, "filter_small": False, "filter_one": False}
    dataset = NF_dataset(mode, filter_option)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    num_batches = len(dataloader)
    model.eval()
    acc_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()
            acc_loss += loss
    test_loss = acc_loss/num_batches
    print(f"Test error: {test_loss}")
    return test_loss

#%% evaluation
def evaluation(loss_df):
    fig, ax = plt.subplots()
    styles = ["-", "--", "--", "--", "--"]
    loss_df.plot(x="epoch", figsize=[10,6], ax=ax, style=styles)

def check_box_plausibility(model):
    """
    input:
        StatBoxEL model
    output:
        the number of implausible boxes
    """
    param_dict = dict()
    for name, param in model.named_parameters():
        param_dict[name] = param
    delta_embedding = param_dict["max_embeddings"] - param_dict["min_embeddings"]
    mask = (delta_embedding<0).any(dim=1)
    return mask.sum().item()


# %% main function
if __name__ == "__main__":
    # current path
    if os.getcwd() != '/workspace/YAGO3_ontology_gen':
        os.chdir('/workspace/YAGO3_ontology_gen/')
    # GPU seeting
    device = "cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # random seed setting
    torch.manual_seed(888)

    # model specification
    size_dict = {"vocab": 135, "relation":6}
    embed_dim = 64
    init_dict = {"min_embed": [100,100+10e-5], "max_embed": [110-10e-5, 110], "rel_scale": [0, 2], "rel_trans": [0, 10-5]}
    # volume_metric options: multi, manhattan, square_sum, euclidean
    volume_metric = "manhattan"
    model = StatBoxEL(size_dict, embed_dim, init_dict, volume_metric).to(device)
    # training specification
    epoch_num = 50
    batch_size = 64
    learning_rate = 0.05
    criterion = MeanAbsoluteError().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # data load
    filter_option = {"filter_zero": False, "filter_small": False, "filter_one": False}
    dataset = NF_dataset("all", filter_option)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # training loop
    train_loss_list = []
    total_loss = []
    NF1_loss = []
    NF2_loss = []
    NF3_loss = []
    NF4_loss = []
    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train(dataloader, model, criterion, optimizer, device)
        train_loss_list.append(train_loss)
        # test seperate loss
        total_loss.append(test("all", model, criterion))
        NF1_loss.append(test("NF1", model, criterion))
        NF2_loss.append(test("NF2", model, criterion))
        NF3_loss.append(test("NF3", model, criterion))
        NF4_loss.append(test("NF4", model, criterion))
    print("Done!")

    loss_df = pd.DataFrame({"Total_loss": total_loss, \
                            "NF1_loss": NF1_loss, \
                            "NF2_loss": NF2_loss, \
                            "NF3_loss": NF3_loss, \
                            "NF4_loss": NF4_loss, \
                            "epoch": np.array(range(epoch_num))+1})
    evaluation(loss_df)
    print(f"Number of implausible boxes: {check_box_plausibility(model)}")
    

# %%
