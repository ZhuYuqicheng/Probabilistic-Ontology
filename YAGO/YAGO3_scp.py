import pandas as pd
from itertools import combinations
# get subclasses of wordnet person
def load_taxonomy(path):
    data = pd.read_csv(path, sep="\t", header=None)
    clean_data = data[[1,2,3]].iloc[1:-1]
    clean_data.columns = ["subclass", "predicate", "superclass"]
    subcls_df = clean_data[clean_data["predicate"]=="rdfs:subClassOf"]
    return subcls_df
def get_person_subcls():
    cls_df = load_taxonomy("./Dataset/yagoSimpleTaxonomy.tsv")
    subcls_df = cls_df[cls_df["superclass"]=="<wordnet_person_100007846>"]
    # only consider worrnet classes
    subcls_df = subcls_df[subcls_df["subclass"].str.contains("wordnet_")]
    # add <wordnet_person_100007846> to list
    subcls_list = list(subcls_df["subclass"])
    subcls_list.append("<wordnet_person_100007846>")
    return subcls_list
def load_type_data(subcls_list):
    count = 0
    chunk_list = []
    for chunk in pd.read_csv("./Dataset/yagoTransitiveTypes.tsv", \
                             sep="\t", header=None, names=["0","instance","predicate","class","literal"], \
                             chunksize = 100000):
        count += 1
        chunk = chunk[["instance", "class"]]
        person_chunk = chunk[chunk["class"].isin(subcls_list)]
        chunk_list.append(person_chunk)
        if count%100 == 0:
            print(f"{count}/2000")
    result = pd.concat(chunk_list, ignore_index=True)
    return result
# query
def count_subclass(type_data, subcls_list):
    mask = type_data["class"].isin(subcls_list)
    filtered_data = type_data[mask]
    return filtered_data.groupby(by="class").count()
def count_interAB(type_data, target_classes):
    df_filtered = type_data[type_data["class"].isin(target_classes)]
    df_counts = df_filtered.groupby("instance")["class"].nunique()
    result = (df_counts==2).sum()
    return result
def run_NF1(type_data, subcls_list):
    subcls_count = count_subclass(type_data, subcls_list)
    target_list = combinations(subcls_list, 2)
    count = []
    result_list = []
    for ind, target_cls in enumerate(target_list):
        num = count_interAB(type_data, target_cls)
        denom = int(subcls_count.loc[target_cls[0]])
        print(f"{ind}/11781: {target_cls} - {num/denom}")
        count.append(num/denom)
        result_list.append(target_cls)
    result = pd.DataFrame({"intersection": result_list, "probability": count})
    result.to_csv("NF1_Person.csv", index=False)
    #return result

if __name__ == "__main__":
    # load data
    subcls_list = get_person_subcls()
    type_data = pd.read_csv("./Dataset/Person_Type.csv")
    run_NF1(type_data, subcls_list)