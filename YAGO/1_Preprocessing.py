import pandas as pd
from itertools import combinations
def get_relation_list():
    schema = pd.read_csv("./Dataset/yagoSchema.tsv",sep="\t",header=None,names=["ID","relation","predicate","range","literal"])
    schema_range = schema[schema["range"]=="<wordnet_person_100007846>"]
    return list(set(schema_range["relation"]))
def get_person_cls():
    cls_df = pd.read_csv("./Dataset/yagoTaxonomy.tsv",sep="\t",header=None,names=["ID","subclass","predicate","class","literal"])
    cls_df_filtered = cls_df[cls_df["class"]=="<wordnet_person_100007846>"]
    result = list(set(cls_df_filtered["subclass"]))
    result.append("<wordnet_person_100007846>")
    return result
def get_fact_data(relation_list):
    print("getting fact data")
    chunk_list = []
    count = 0
    for chunk in pd.read_csv("./Dataset/yagoFacts.tsv",sep="\t",header=None,names=["ID","subject","predicate","object","literal"], chunksize=100000):
        chunk_list.append(chunk[chunk["predicate"].isin(relation_list)])
        count += 1
        print(f"{count}/149")
    return pd.concat(chunk_list, ignore_index=True)
def get_type_instance():
    print("getting type instances")
    chunk_list = []
    count = 0
    for chunk in pd.read_csv("./Dataset/yagoTransitiveTypes.tsv",sep="\t",header=None,names=["ID","subject","predicate","object","literal"], chunksize=100000):
        chunk_list += list(chunk["subject"])
        count += 1
        print(f"{count}/2049")
    return set(chunk_list)
def get_intersect_instance(fact_data, tran_instance_set):
    result = set(list(fact_data["subject"])+list(fact_data["object"]))&tran_instance_set
    return list(result)
def get_final_Type(instance_list, person_cls):
    print("getting final type data...")
    chunk_list = []
    count = 0
    for chunk in pd.read_csv("./Dataset/yagoTransitiveTypes.tsv",sep="\t",header=None,names=["ID","subject","predicate","object","literal"], chunksize=100000):
        mask = (chunk["subject"].isin(instance_list))&(chunk["object"].isin(person_cls))
        chunk_filter = chunk[mask]
        chunk_list.append(chunk_filter[["subject", "object"]])
        count += 1
        print(f"{count}/2040")
    result = pd.concat(chunk_list, ignore_index=True)
    result.to_csv("Person_Types.csv", index=None)

if __name__ == "__main__":
    # get instances both in yagoFacts and yagoTransitiveTypes
    fact_data = pd.read_csv("Person_Facts.csv")
    tran_instance_set = get_type_instance()
    instance_list = get_intersect_instance(fact_data, tran_instance_set) # instance_list (all instances)
    # get subclasses of wordnet_Person from yagoTaxonomy.tsv
    person_cls = get_person_cls()
    get_final_Type(instance_list, person_cls)