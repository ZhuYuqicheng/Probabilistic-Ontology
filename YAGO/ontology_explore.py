#%%
import os
import pandas as pd
from itertools import combinations
#%%
def load_data(path):
    subsumption = pd.read_csv(path, sep="\t", header=None)
    subsumption = subsumption.iloc[:,1:-1]
    subsumption.columns = ["subClass", "superClass"]
    superclass = list(subsumption["superClass"].unique())
    return subsumption, superclass
def load_class():
    subsumption_list = []
    superclass_list = []
    path_list = ['subClassOf/RelationsExtractor.txt',
                    'subClassOf/WordNetLinks.txt',
                    'subClassOf/ConceptLinker.txt']
    for file_path in path_list:
        subsumption, superclass = load_data(file_path)
        superclass_list += superclass
        subsumption_list.append(subsumption)
    subsumption_list = pd.concat(subsumption_list, ignore_index=True)
    return subsumption_list, superclass_list
def load_instance():
    type_data_artical = pd.read_csv('type/ArticleExtractor.txt', sep="\t", header=None)
    type_data_checked = pd.read_csv('type/CheckedFactExtractor.txt', sep="\t", header=None)
    type_data_isa = pd.read_csv('type/IsAExtractor.txt', sep="\t", header=None)
    result = pd.concat([type_data_artical, type_data_checked, type_data_isa], ignore_index=True)
    result = result.iloc[:,1:-1]
    result.columns = ["instance", "class"]
    return result
# query functions
def subclass_counting(data):
    return data.groupby(by="superClass").count().sort_values(by="subClass", ascending=False)
def instance_counting(data):
    return data.groupby(by="class").count().sort_values(by="instance", ascending=False)
def query_subclass(subsumption_list, query_class):
    """including the superclass"""
    mask = subsumption_list["superClass"] == query_class
    result = subsumption_list[mask]["subClass"].to_list()
    result.append(query_class)
    return result
def query_count(type_data, query_class):
    return (type_data["class"] == query_class).sum()

def count_subclass(cls):
    sub_Person_list = query_subclass(subsumption_list, cls)
    mask = type_data["class"].isin(sub_Person_list)
    filtered_data = type_data[mask]
    result = instance_counting(filtered_data)
    return result
def count_interAB(target_classes):
    df_filtered = type_data[type_data["class"].isin(target_classes)]
    df_counts = df_filtered.groupby("instance")["class"].nunique()
    result = (df_counts==2).sum()
    return result

# %%
subsumption_list, superclass_list = load_class()
type_data = load_instance()

# %%
target_supercls = "wordnet_person_100007846"
subcls_count = count_subclass(target_supercls)
#subcls_list = subcls_count.index.to_list()[:10]
subcls_count_filter = subcls_count[subcls_count["instance"]>100]
subcls_list = subcls_count_filter.index.to_list()
target_list = combinations(subcls_list, 2)
count = []
result_list = []
for ind, target_cls in enumerate(target_list):
    n = count_interAB(target_cls)
    d = int(subcls_count.loc[target_cls[0]])
    print(f"{ind}: {target_cls} - {n/d}")
    count.append(n/d)
    result_list.append(target_cls)
result = pd.DataFrame({"intersection": result_list, "probability": count})
result.to_csv("NF1_Person.csv", index=False)
#target_classes = ("wordnet_person_100007846", "wordnet_leader_109623038")

# %%
