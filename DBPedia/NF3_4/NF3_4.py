#%%
from SPARQLWrapper import SPARQLWrapper, JSON
from itertools import combinations
import pandas as pd

def get_target_class():
    interest_list = ["Person", "Organisation", "Place", "Work", "Event", "Species", "Activity", "Food", "Device"]
    prefix = "http://dbpedia.org/ontology/"
    return [prefix+cls for cls in interest_list]
def intersect_rC(sparql):
    cls_list = get_target_class()
    result = dict()
    for cls in cls_list:
        query = """SELECT DISTINCT ?predicate (COUNT(?x) AS ?number) """
        query += f" WHERE{{?y a <{cls}> . ?x ?predicate ?y}}"
        sparql.setQuery(query)
        ret = sparql.queryAndConvert()
        for x in ret["results"]["bindings"]:
            result[x["predicate"]["value"]+"+"+cls] = int(x["number"]["value"])
    return result
def intersect_rAB(lookup, sparql, A, B):
    """r(x,y) and A(y) -> B(x)"""
    query = """SELECT DISTINCT ?predicate (COUNT(?x) AS ?number) """
    query += f" WHERE{{?y a <{A}> . ?x a <{B}> ."
    query += f" ?x ?predicate ?y }}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    result = []
    for x in ret["results"]["bindings"]:
        denom = lookup[x["predicate"]["value"]+"+"+A]
        result.append((x["predicate"]["value"], A, B, int(x["number"]["value"])/denom))
    return pd.DataFrame(result, columns=["Predicate(x,y)", "ConceptA(y)", "ConceptB(x)", "Probability"])
def type3(sparql):
    lookup_dict = intersect_rC(sparql)
    comb = list(combinations(get_target_class(), 2))
    result = []
    for A,B in comb:
        print(f"Current: {A} and {B}")
        result.append(intersect_rAB(lookup_dict, sparql, A, B))
    result = pd.concat(result, ignore_index=True)
    result.to_csv("NF3.csv", index=False)

def count_B(sparql):
    cls_list = get_target_class()
    result = dict()
    for cls in cls_list:
        query = """SELECT DISTINCT (COUNT(?x) AS ?number) """
        query += f" WHERE{{?x a <{cls}>}}"
        sparql.setQuery(query)
        ret = sparql.queryAndConvert()
        for x in ret["results"]["bindings"]:
            result[cls] = int(x["number"]["value"])
    return result
def intersect_BrA(lookup, sparql, A, B):
    """r(x,y) and A(y) -> B(x)"""
    query = """SELECT DISTINCT ?predicate (COUNT(?x) AS ?number) """
    query += f" WHERE{{?y a <{A}> . ?x a <{B}> ."
    query += f" ?x ?predicate ?y }}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    result = []
    for x in ret["results"]["bindings"]:
        denom = lookup[B]
        if int(x["number"]["value"])/denom < 1:
            result.append((B, x["predicate"]["value"], A, int(x["number"]["value"])/denom))
    return pd.DataFrame(result, columns=["ConceptB(x)", "Predicate(x,y)", "ConceptA(y)", "Probability"])
def type4(sparql):
    lookup_dict = count_B(sparql)
    comb = list(combinations(get_target_class(), 2))
    result = []
    for A,B in comb:
        print(f"Current: {A} and {B}")
        result.append(intersect_BrA(lookup_dict, sparql, A, B))
    result = pd.concat(result, ignore_index=True)
    result.to_csv("NF4.csv", index=False)

#%%
if __name__ == "__main__":
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    
    #type3(sparql)
    type4(sparql)

# %%
