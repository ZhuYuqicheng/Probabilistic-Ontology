#%%
import pandas as pd
from itertools import combinations
from itertools import permutations
def get_count_A(type_data):
    group_df = type_data.groupby(by="object").count().sort_values(by="subject", ascending=False)
    cls_name = group_df.index.to_list()
    count = group_df["subject"].to_list()
    result = pd.DataFrame({"class":cls_name, "count":count})
    result.to_csv("count_A.csv", index=None)

def count_intersec(type_data, target_classes):
    df_filtered = type_data[type_data["object"].isin(target_classes)]
    df_counts = df_filtered.groupby("subject")["object"].nunique()
    result = (df_counts==len(target_classes)).sum()
    return result
def get_count_AB(type_data):
    # get class list
    group_df = type_data.groupby(by="object").count().sort_values(by="subject", ascending=False)
    group_df = group_df[group_df["subject"]>1]
    subcls_list = group_df.index.to_list()
    # count loop
    count = []
    concept1 = []
    concept2 = []
    target_list = combinations(subcls_list, 2)
    for ind, target_cls in enumerate(target_list):
        num = count_intersec(type_data, target_cls)
        print(f"countAB({ind}/8000): {target_cls}, count: {num}")
        count.append(num)
        concept1.append(target_cls[0])
        concept2.append(target_cls[1])
    result = pd.DataFrame({"Concept1": concept1, "Concept2": concept2, "count": count})
    result.to_csv("count_AB.csv", index=False)
def get_count_ABC(type_data):
    # get class list
    group_df = type_data.groupby(by="object").count().sort_values(by="subject", ascending=False)
    group_df = group_df[:50]
    subcls_list = group_df.index.to_list()
    # count loop
    count = []
    concept1 = []
    concept2 = []
    concept3 = []
    target_list = combinations(subcls_list, 3)
    for ind, target_cls in enumerate(target_list):
        num = count_intersec(type_data, target_cls)
        print(f"countABC({ind}/19600): {target_cls}, count: {num}")
        count.append(num)
        concept1.append(target_cls[0])
        concept2.append(target_cls[1])
        concept3.append(target_cls[2])
    result = pd.DataFrame({"Concept1": concept1, "Concept2": concept2, "Concept3": concept3, "count": count})
    result.to_csv("count_ABC.csv", index=False)
def asign_class2instance(fact_data, type_data):
    subject_cls_list = []
    object_cls_list = []
    for ind in range(len(fact_data)):
        print(f"{ind}/{len(fact_data)}")
        subject = fact_data.loc[ind]["subject"]
        object = fact_data.loc[ind]["object"]
        subject_cls = type_data["object"][type_data["subject"]==subject].to_list()
        object_cls = type_data["object"][type_data["subject"]==object].to_list()
        subject_cls_list.append(subject_cls)
        object_cls_list.append(object_cls)
    cls_list = pd.DataFrame({"subject_class":subject_cls_list, "object_class":object_cls_list})
    result = pd.concat([fact_data[["subject", "object"]], cls_list], axis=1)
    result.to_pickle("wordnet_fact_with_class.pkl")

# NF3,4
def get_cls_instance(type_data):
    # calculate subclass list
    group_df = type_data.groupby(by="object").count().sort_values(by="subject", ascending=False)
    group_df = group_df[group_df["subject"]>1]
    subcls_list = group_df.index.to_list()
    # find corresponding instance list for each subclass
    cls_instance = dict()
    for cls in subcls_list:
        cls_instance[cls] = list(type_data[type_data["object"]==cls]["subject"].unique())
    return cls_instance
def query_rB(fact_data, cls_instance_dict, query_relation, query_class):
    filtered_fact = fact_data[fact_data["predicate"]==query_relation]
    mask = filtered_fact["object"].isin(cls_instance_dict[query_class])
    result_fact = filtered_fact[mask]
    return len(result_fact)
def query_ArB(fact_data, cls_instance_dict, query_relation, query_classA, query_classB):
    filtered_fact = fact_data[fact_data["predicate"]==query_relation]
    mask = filtered_fact["subject"].isin(cls_instance_dict[query_classA]) & filtered_fact["object"].isin(cls_instance_dict[query_classB])
    result_fact = filtered_fact[mask]
    return len(result_fact)
def get_count_rA(type_data, fact_data, cls_instance_dict):
    # get relation
    relation_list = list(fact_data["predicate"].unique())
    # get class list
    group_df = type_data.groupby(by="object").count().sort_values(by="subject", ascending=False)
    group_df = group_df[group_df["subject"]>1]
    class_list = group_df.index.to_list()
    # query count
    count = 0
    query_rel_list = []
    query_cls_list = []
    query_count_list = []
    for query_relation in relation_list:
        for query_class in class_list:
            count += 1
            query_rel_list.append(query_relation)
            query_cls_list.append(query_class)
            query_count = query_rB(fact_data, cls_instance_dict, query_relation, query_class)
            query_count_list.append(query_count)
            print(f"{count}/{len(relation_list)*len(class_list)}: {query_relation} and {query_class} -> {query_count}")
    result = pd.DataFrame({"relation":query_rel_list, "class":query_cls_list, "count":query_count_list})
    result.to_csv("count_rA.csv", index=False)
def get_count_rAB(type_data, fact_data, cls_instance_dict):
    # get relation
    relation_list = list(fact_data["predicate"].unique())
    # get class list
    group_df = type_data.groupby(by="object").count().sort_values(by="subject", ascending=False)
    group_df = group_df[group_df["subject"]>1]
    class_list = group_df.index.to_list()
    # query count
    count = 0
    query_rel_list = []
    query_cls1_list = []
    query_cls2_list = []
    query_count_list = []
    for query_relation in relation_list:
        for query_class1, query_class2 in permutations(class_list, 2):
            count += 1
            query_rel_list.append(query_relation)
            query_cls1_list.append(query_class1)
            query_cls2_list.append(query_class2)
            query_count = query_ArB(fact_data, cls_instance_dict, query_relation, query_class1, query_class2)
            query_count_list.append(query_count)
            print(f"{count}/{len(relation_list)*16002}: {query_relation} and {query_class1} and {query_class2} -> {query_count}")
    result = pd.DataFrame({"classA":query_cls1_list, "relation":query_rel_list, "classB":query_cls2_list, "count":query_count_list})
    result.to_csv("count_rAB.csv", index=False)
#%% main
if __name__ == "__main__":
    # load data
    type_data = pd.read_csv("wordnet_person_types.csv")
    fact_data = pd.read_csv("wordnet_person_facts.csv")
    #get_count_A(type_data)
    #get_count_AB(type_data)
    #get_count_ABC(type_data)
    # asign_class2instance(fact_data, type_data) # too expensive
    cls_instance_dict = get_cls_instance(type_data)
    #get_count_rA(type_data, fact_data, cls_instance_dict)
    get_count_rAB(type_data, fact_data, cls_instance_dict)