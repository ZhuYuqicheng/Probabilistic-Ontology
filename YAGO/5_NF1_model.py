# %% package
import pandas as pd
import torch
from torch import nn
from torch.distributions import uniform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn import preprocessing

#%% Data processing
def import_data(device, split=False, filter=False):
    """
    input:
        device: specify the GPU device
        split: wether split the data to train and test
        filter: wether filter out disjointness information
    output:
        (X_dict, y)
        X_dict: contains input tensor (already transfer to GPU) for different NFs
        y: concatenated probabilities for all NFs (already transfer to GPU)
    """
    if filter and split:
        pass
    # create lookup table for ontology classes
    class_table = pd.read_csv("count_A.csv")
    le = preprocessing.LabelEncoder()
    le.fit(class_table["class"].unique())

    NF1_data = pd.read_csv("./YAGO3_result/result_NF1.csv")
    NF1_X = torch.tensor(NF1_data[["ConceptA", "ConceptB"]].apply(le.transform).values).to(device)
    NF1_y = torch.tensor(NF1_data["Probability"]).to(device)

    #NF2_data = pd.read_csv("./YAGO3_result/result_NF2.csv")
    #NF2_X = torch.tensor(NF2_data[["ConceptA", "ConceptB", "ConceptC"]].apply(le.transform).values).to(device)

    #NF3_data = pd.read_csv("./YAGO3_result/result_NF3.csv")
    #NF4_data = pd.read_csv("./YAGO3_result/result_NF4.csv")

    #y_concat = pd.concat([NF1_data["Probability"], NF2_data["Probability"]], ignore_index=True)
    #y = torch.tensor(y_concat.values).to(device)
    
    return (NF1_X, NF1_y)

class NF1_dataset(Dataset):
    def __init__(self) -> None:
        class_table = pd.read_csv("count_A.csv")
        le = preprocessing.LabelEncoder()
        le.fit(class_table["class"].unique())
        row_data = pd.read_csv("./YAGO3_result/result_NF1.csv")
        self.X = torch.tensor(row_data[["ConceptA", "ConceptB"]].apply(le.transform).values)
        self.y = torch.tensor(row_data["Probability"])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])
    
class NF2_dataset(Dataset):
    def __init__(self) -> None:
        class_table = pd.read_csv("count_A.csv")
        le = preprocessing.LabelEncoder()
        le.fit(class_table["class"].unique())
        row_data = pd.read_csv("./YAGO3_result/result_NF2.csv")
        self.X = torch.tensor(row_data[["ConceptA", "ConceptB", "ConceptC"]].apply(le.transform).values)
        self.y = torch.tensor(row_data["Probability"])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

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
    def __init__(self, size_dict, embed_dim, init_dict) -> None:
        super(StatBoxEL, self).__init__()
        min_embeddings = self.init_embedding(size_dict["vocab"], embed_dim, init_dict["min_embed"])
        max_embeddings = self.init_embedding(size_dict["vocab"], embed_dim, init_dict["max_embed"])
        relation_embeddings = self.init_embedding(size_dict["relation"], embed_dim, init_dict["relation"])

        self.min_embeddings = nn.Parameter(min_embeddings)
        self.max_embeddings = nn.Parameter(max_embeddings)
        self.relation_embeddings = nn.Parameter(relation_embeddings)

    def forward(self, x):
        # NF1
        #NF1_data = x["NF1"]
        # class_index_A = NF1_data[:,0]
        # NF1_min_A = torch.index_select(self.min_embeddings, 0, class_index_A)
        # NF1_max_A = torch.index_select(self.max_embeddings, 0, class_index_A)
        # class_index_B = NF1_data[:,1]
        # NF1_min_B = torch.index_select(self.min_embeddings, 0, class_index_B)
        # NF1_max_B = torch.index_select(self.max_embeddings, 0, class_index_B)

        # NF1_class_A = Box(NF1_min_A, NF1_max_A)
        # NF1_class_B = Box(NF1_min_B, NF1_max_B)

        # class_AB = self.intersection(NF1_class_A, NF1_class_B)
        # pred_p = self.volumes(class_AB)/self.volumes(NF1_class_A)
        #NF1_pred = self.NF1_pred(NF1_data)

        # NF2
        #NF2_data = x["NF2"]
        #NF2_pred = self.NF2_pred(NF2_data)

        # NF3
        # NF4

        NF1_pred = self.NF1_pred(x)

        #return torch.cat((NF1_pred, NF2_pred), 0)
        return NF1_pred

    def init_embedding(self, vocab_size:int, embed_dim:int, init_value:list):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        embedding = distribution.sample((vocab_size, embed_dim))
        return embedding
    
    def volumes(self, box):
        return box.delta_embed.prod(1, keepdim=True)
    
    def intersection(self, class_A, class_B):
        min_embed = torch.max(class_A.min_embed, class_B.min_embed)
        max_embed = torch.min(class_A.max_embed, class_B.max_embed)
        intersected_box = Box(min_embed, max_embed)
        return intersected_box
    def relation_mapping(self, relation, class_A):
        pass

    def NF1_pred(self, NF1_data):
        class_index_A = NF1_data[:,0]
        NF1_min_A = torch.index_select(self.min_embeddings, 0, class_index_A)
        NF1_max_A = torch.index_select(self.max_embeddings, 0, class_index_A)
        class_index_B = NF1_data[:,1]
        NF1_min_B = torch.index_select(self.min_embeddings, 0, class_index_B)
        NF1_max_B = torch.index_select(self.max_embeddings, 0, class_index_B)

        NF1_class_A = Box(NF1_min_A, NF1_max_A)
        NF1_class_B = Box(NF1_min_B, NF1_max_B)

        class_AB = self.intersection(NF1_class_A, NF1_class_B)
        pred_p = self.volumes(class_AB)/self.volumes(NF1_class_A)
        return pred_p
    
    def NF2_pred(self, NF2_data):
        class_index_A1 = NF2_data[:,0]
        NF2_min_A1 = torch.index_select(self.min_embeddings, 0, class_index_A1)
        NF2_max_A1 = torch.index_select(self.max_embeddings, 0, class_index_A1)
        class_index_A2 = NF2_data[:,1]
        NF2_min_A2 = torch.index_select(self.min_embeddings, 0, class_index_A2)
        NF2_max_A2 = torch.index_select(self.max_embeddings, 0, class_index_A2)
        class_index_B = NF2_data[:,2]
        NF2_min_B = torch.index_select(self.min_embeddings, 0, class_index_B)
        NF2_max_B = torch.index_select(self.max_embeddings, 0, class_index_B)

        class_A1 = Box(NF2_min_A1, NF2_max_A1)
        class_A2 = Box(NF2_min_A2, NF2_max_A2)
        class_B = Box(NF2_min_B, NF2_max_B)

        class_A1A2 = self.intersection(class_A1, class_A2)
        class_A1A2B = self.intersection(class_A1A2, class_B)
        pred_p = self.volumes(class_A1A2B)/self.volumes(class_A1A2)
        return pred_p
    
    def NF3_loss(self, class_A, relation, class_B, p):
        class_mapping = self.relation_mapping(relation, class_B)
        class_ArB = self.intersection(class_A, class_mapping)
        pred_p = self.volumes(class_ArB)/self.volumes(class_A)
        loss = abs(pred_p - p)
        return loss
    def NF4_loss(self, relation, class_A, class_B, p):
        class_mapping = self.relation_mapping(relation, class_A)
        class_ArB = self.intersection(class_mapping, class_B)
        pred_p = self.volumes(class_ArB)/self.volumes(class_mapping)
        loss = abs(pred_p - p)
        return loss

def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Forward propagation
        pred = model(X)
        # Loss calculation
        loss = criterion(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # monitoring
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Training Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return torch.Tensor.cpu(loss).detach().numpy()

def test(data, model, criterion):
    model.eval()
    with torch.no_grad():
        pred = model(data[0])
        test_loss = criterion(pred, data[1]).item()
    print(f"Test error: {test_loss}")

def evaluation(loss_list, epoch_num):
    plt.plot(range(epoch_num), loss_list)
    plt.show()


# %% main function
if __name__ == "__main__":
    # GPU seeting
    device = "cuda:3" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # random seed setting
    torch.manual_seed(888)

    # model specification
    size_dict = {"vocab": 135, "relation":6}
    embed_dim = 8
    init_dict = {"min_embed": [0,0.1], "max_embed": [0.7, 1], "relation": [0, 1]}
    model = StatBoxEL(size_dict, embed_dim, init_dict).to(device)
    # training specification
    epoch_num = 30
    batch_size = 64
    learning_rate = 0.05
    criterion = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # data load
    #df = pd.DataFrame({"ConceptA": [0,1,2,1,0,0], "ConceptB": [1,0,0,0,2,1], "prob": [0.8,0.7,0.6,0.7,1.0,1.0]})
    #train_data = torch.tensor(df[["ConceptA", "ConceptB"]].values).to(device)
    #train_label = torch.tensor(df["prob"].values).to(device)
    #data = (train_data, train_label)
    data = import_data(device)
    NF1_data = NF1_dataset()
    NF1_dataloader = DataLoader(NF1_data, batch_size=batch_size, shuffle=True)

    # training loop
    train_loss_list = []
    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train(NF1_dataloader, model, criterion, optimizer)
        train_loss_list.append(train_loss)
        #test(data, model, criterion)
    print("Done!")
    evaluation(train_loss_list, epoch_num)

# %%
