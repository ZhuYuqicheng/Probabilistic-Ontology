import pandas as pd
from itertools import permutations
def load_count_A():
    count_A = pd.read_csv("count_A.csv")
    count_A.columns = ["ConceptA", "A_count"]
    return count_A
def load_count_AB():
    """A(x) -> B(x)"""
    count_AB = pd.read_csv("count_AB.csv")
    count_AB.columns = ["ConceptA", "ConceptB", "AB_count"]
    count_BA = pd.DataFrame({"ConceptA": count_AB["ConceptB"], "ConceptB": count_AB["ConceptA"], "AB_count": count_AB["AB_count"]})
    final_count_AB = pd.concat([count_AB, count_BA], ignore_index=True)
    return final_count_AB
def load_count_ABC():
    """A(x) and B(x) -> C(x)"""
    count_ABC = pd.read_csv("count_ABC.csv")
    count_ABC.columns = ["ConceptA", "ConceptB", "ConceptC", "ABC_count"]
    concat_list = []
    for concept_A, concept_B, concept_C in permutations(["ConceptA", "ConceptB", "ConceptC"], 3):
        concat_list.append(\
            pd.DataFrame({"ConceptA": count_ABC[concept_A], \
                          "ConceptB": count_ABC[concept_B], \
                            "ConceptC": count_ABC[concept_C], \
                            "ABC_count": count_ABC["ABC_count"]}))
    return pd.concat(concat_list, ignore_index=True)
def load_count_rA():
    """r(x,y) and A(y)"""
    count_rA = pd.read_csv("count_rA.csv")
    count_rA.columns = ["relation", "ConceptB", "rA_count"]
    return count_rA
def load_count_rAB():
    """
    A(x) -> r(x,y) and B(y)
    or
    r(x,y) and B(y) -> A(x)
    """
    count_rAB = pd.read_csv("count_rAB.csv")
    count_rAB.columns = ["ConceptA", "relation", "ConceptB", "rAB_count"]
    return count_rAB

def get_NF1_df():
    count_A = load_count_A()
    count_AB = load_count_AB()
    merge_df = pd.merge(count_AB, count_A, how="inner")
    result = merge_df.assign(Probability=lambda x: x.AB_count/x.A_count)
    return result
def get_NF2_df():
    count_AB = load_count_AB()
    count_ABC = load_count_ABC()
    merge_df = pd.merge(count_ABC, count_AB, how="inner")
    result = merge_df.assign(Probability=lambda x: x.ABC_count/x.AB_count)
    return result
def get_NF3_df():
    count_A = load_count_A()
    count_rAB = load_count_rAB()
    merge_df = pd.merge(count_rAB, count_A, how="inner")
    result = merge_df.assign(Probability=lambda x: x.rAB_count/x.A_count)
    return result
def get_NF4_df():
    count_rA = load_count_rA()
    count_rAB = load_count_rAB()
    merge_df = pd.merge(count_rAB, count_rA, how="inner")
    result = merge_df.assign(Probability=lambda x: x.rAB_count/x.rA_count)
    return result

if __name__ == "__main__":
    NF1_data = get_NF1_df()
    NF1_data.to_csv("result_NF1.csv", index=False)

    NF2_data = get_NF2_df()
    NF2_data.to_csv("result_NF2.csv", index=False)

    NF3_data = get_NF3_df()
    NF3_data.to_csv("result_NF3.csv", index=False)

    NF4_data = get_NF4_df()
    NF4_data.to_csv("result_NF4.csv", index=False)