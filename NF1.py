#%%
from SPARQLWrapper import SPARQLWrapper, JSON
from itertools import combinations
import pandas as pd

def get_target_class(sparql):
    query = """SELECT DISTINCT ?subclass """
    query += f" WHERE{{?subclass rdfs:subClassOf owl:Thing}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    ret_list = [x["subclass"]["value"] for x in ret["results"]["bindings"]]
    return ret_list[:58]
def get_loop_class(sparql):
    subcls_list = get_target_class(sparql)
    ret_list = []
    for subcls in subcls_list:
        query = """SELECT DISTINCT ?subclass """
        query += f" WHERE{{?subclass rdfs:subClassOf* <{subcls}>}}"
        sparql.setQuery(query)
        ret = sparql.queryAndConvert()
        ret_list += [x["subclass"]["value"] for x in ret["results"]["bindings"]]
    return ret_list
def single_class_result(sparql, cls_str):
    # count A and B
    subcls_query = """SELECT DISTINCT ?subclass (COUNT(?instance) AS ?number)"""
    subcls_query += f" WHERE{{?subclass rdfs:subClassOf <{cls_str}> . "
    subcls_query += f"?instance a ?subclass . ?instance a <{cls_str}>}}"
    sparql.setQuery(subcls_query)
    subclass_out = sparql.queryAndConvert()
    # count A
    cls_query = """SELECT DISTINCT (COUNT(?instance) AS ?number)"""
    cls_query += f" WHERE{{?instance a <{cls_str}>}}"
    sparql.setQuery(cls_query)
    class_out = sparql.queryAndConvert()
    denominator = int(class_out["results"]["bindings"][0]["number"]["value"])
    # generate result
    result = []
    for x in subclass_out["results"]["bindings"]:
        numerator = int(x["number"]["value"])
        if (cls_str != x["subclass"]["value"]) & (numerator/denominator<1.0):
            result.append((cls_str, x["subclass"]["value"], numerator/denominator))
    return pd.DataFrame(result, columns=["ConceptA", "ConceptB", "Probability"])

#%%
if __name__ == "__main__":
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    loop_class = get_loop_class(sparql)
    result = []
    for cls in loop_class:
        result.append(single_class_result(sparql, cls))
        print(cls)
    result_pd = pd.concat(result, ignore_index=True)
    result_pd.to_csv("NF1.csv", index=False)
# %%
