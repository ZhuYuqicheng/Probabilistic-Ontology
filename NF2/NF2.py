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
def AintersectB(sparql, cls):
    query = """SELECT DISTINCT ?subclass1 ?subclass2 (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass1 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass2 rdfs:subClassOf <{cls}> ."
    query += f" ?instance a ?subclass1 . ?instance a ?subclass2 ."
    query += f" FILTER (?subclass1 != ?subclass2)}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    result = dict()
    for x in ret["results"]["bindings"]:
        result[x["subclass1"]["value"]+"+"+x["subclass2"]["value"]] = x["number"]["value"]
    query = """SELECT DISTINCT ?subclass ?subsub (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass rdfs:subClassOf <{cls}> ."
    query += f" ?subsub rdfs:subClassOf ?subclass ."
    query += f" ?instance a ?subclass . ?instance a ?subsub .}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    for x in ret["results"]["bindings"]:
        result[x["subclass"]["value"]+"+"+x["subsub"]["value"]] = x["number"]["value"]
    return result



def triple_intersection(sparql, cls):
    # ABC intersection
    query = """SELECT DISTINCT ?subclass1 ?subclass2 ?subclass3 (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass1 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass2 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass3 rdfs:subClassOf <{cls}> ."
    query += f" FILTER (?subclass1 != ?subclass2) ."
    query += f" FILTER (?subclass1 != ?subclass3) ."
    query += f" FILTER (?subclass2 != ?subclass3) ."
    query += f" ?instance a ?subclass1 . ?instance a ?subclass2 . ?instance a ?subclass3}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    result = []
    for x in ret["results"]["bindings"]:
        result.append((x["subclass1"]["value"],x["subclass2"]["value"],x["subclass3"]["value"],x["number"]["value"]))
    # A_BC intersection
    query = """SELECT DISTINCT ?subcalss ?subsub1 ?subsub2 (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass rdfs:subClassOf <{cls}> ."
    query += f" ?subsub1 rdfs:subClassOf ?subclass ."
    query += f" ?subsub2 rdfs:subClassOf ?subclass ."
    query += f" FILTER (?subsub1 != ?subsub2) ."
    query += f" ?instance a ?subclass . ?instance a ?subsub1 . ?instance a ?subsub2}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    for x in ret["results"]["bindings"]:
        result.append((x["subclass"]["value"],x["subsub1"]["value"],x["subsub2"]["value"], x["number"]["value"]))
    # AB_C intersection
    query = """SELECT DISTINCT ?subclass1 ?subclass2 ?subsub (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass1 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass2 rdfs:subClassOf <{cls}> ."
    query += f" ?subsub rdfs:subClassOf ?subclass1 ."
    query += f" FILTER (?subclass1 != ?subclass2) ."
    query += f" ?instance a ?subclass1 . ?instance a ?subclass2 . ?instance a ?subsub}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    for x in ret["results"]["bindings"]:
        result.append((x["subclass1"]["value"],x["subclass2"]["value"],x["subsub"]["value"],x["number"]["value"]))

    return pd.DataFrame(result, columns=["ConceptA", "ConceptB", "ConceptC", "Count"])

#%% core code
def intersectAB(sparql, cls):
    query = """SELECT DISTINCT ?subclass1 ?subclass2 (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass1 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass2 rdfs:subClassOf <{cls}> ."
    query += f" FILTER (?subclass1 != ?subclass2) ."
    query += f" ?instance a ?subclass1 . ?instance a ?subclass2 }}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    result = dict()
    for x in ret["results"]["bindings"]:
        result[x["subclass1"]["value"]+"+"+x["subclass2"]["value"]] = int(x["number"]["value"])/4
    return result
def lookup_AB_C(result, sparql, cls):
    query = """SELECT DISTINCT ?subclass ?subsub (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass rdfs:subClassOf <{cls}> ."
    query += f" ?subsub rdfs:subClassOf ?subclass ."
    query += f" ?instance a ?subclass . ?instance a ?subsub }}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    for x in ret["results"]["bindings"]:
        result[x["subclass"]["value"]+"+"+x["subsub"]["value"]] = int(x["number"]["value"])/4
    return result

def intersectABC(sparql, cls):
    query = """SELECT DISTINCT ?subclass1 ?subclass2 ?subclass3 (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass1 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass2 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass3 rdfs:subClassOf <{cls}> ."
    query += f" FILTER (?subclass1 != ?subclass2) ."
    query += f" FILTER (?subclass1 != ?subclass3) ."
    query += f" FILTER (?subclass2 != ?subclass3) ."
    query += f" ?instance a ?subclass1 . ?instance a ?subclass2 . ?instance a ?subclass3}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    result = []
    for x in ret["results"]["bindings"]:
        result.append((x["subclass1"]["value"],x["subclass2"]["value"],x["subclass3"]["value"],int(x["number"]["value"])/8))
    return pd.DataFrame(result, columns=["ConceptA", "ConceptB", "ConceptC", "Count"])
def intersectAB_C(sparql, cls):
    query = """SELECT DISTINCT ?subclass1 ?subclass2 ?subsub (COUNT(?instance) AS ?number) """
    query += f" WHERE{{?subclass1 rdfs:subClassOf <{cls}> ."
    query += f" ?subclass2 rdfs:subClassOf <{cls}> ."
    query += f" ?subsub rdfs:subClassOf ?subclass1 ."
    query += f" FILTER (?subclass1 != ?subclass2) ."
    query += f" ?instance a ?subclass1 . ?instance a ?subclass2 . ?instance a ?subsub}}"
    sparql.setQuery(query)
    ret = sparql.queryAndConvert()
    result = []
    for x in ret["results"]["bindings"]:
        result.append((x["subclass1"]["value"],x["subclass2"]["value"],x["subsub"]["value"],int(x["number"]["value"])/8))
    return pd.DataFrame(result, columns=["ConceptA", "ConceptB", "ConceptC", "Count"])


def get_cls_name(iri):
    return iri.split("/")[-1]
def query_count(lookup, sample, str1, str2):
    if sample[str1]+"+"+sample[str2] in lookup:
        return lookup[sample[str1]+"+"+sample[str2]]
    elif sample[str2]+"+"+sample[str1] in lookup:
        return lookup[sample[str2]+"+"+sample[str1]]
    else:
        query = """SELECT DISTINCT (COUNT(?x) AS ?number) """
        query += f" WHERE{{?x a <{sample[str1]}> . ?x a <{sample[str2]}>}}"
        sparql.setQuery(query)
        ret = sparql.queryAndConvert()
        return int(ret["results"]["bindings"][0]["number"]["value"])
def compute_prob(lookup, sample):
    result = []
    denom = query_count(lookup, sample, "ConceptA", "ConceptB")
    if denom != 0:
        prob = sample["Count"]/denom
        result.append((sample["ConceptA"], sample["ConceptB"], sample["ConceptC"], prob))
    denom = query_count(lookup, sample, "ConceptA", "ConceptC")
    if denom != 0:
        prob = sample["Count"]/denom
        result.append((sample["ConceptA"], sample["ConceptC"], sample["ConceptB"], prob))
    denom = query_count(lookup, sample, "ConceptB", "ConceptC")
    if denom != 0:
        prob = sample["Count"]/denom
        result.append((sample["ConceptB"], sample["ConceptC"], sample["ConceptA"], prob))
    return pd.DataFrame(result, columns=["ConceptA1", "ConceptA2", "ConceptB", "Probability"])

def type2_interABC(sparql):
    loop_class = get_loop_class(sparql)
    result = []
    for cls in loop_class:
        print(cls)
        lookup_dict = intersectAB(sparql, cls)
        df = intersectABC(sparql, cls)
        sample_result = []
        for _, sample in df.iterrows():
            sample_result.append(compute_prob(lookup_dict, sample))
        if len(sample_result) != 0:
            result.append(pd.concat(sample_result, ignore_index=True))
    result_pd = pd.concat(result, ignore_index=True)
    result_pd.to_csv("NF2_interABC.csv", index=False) 
def type2_interAB_C(sparql):
    loop_class = get_loop_class(sparql)
    result = []
    count = 0
    for cls in loop_class:
        count += 1
        print(f"{count}: {cls}")
        lookup_dict = lookup_AB_C(intersectAB(sparql, cls), sparql, cls)
        df = intersectAB_C(sparql, cls)
        sample_result = []
        for _, sample in df.iterrows():
            sample_result.append(compute_prob(lookup_dict, sample))
        if len(sample_result) != 0:
            result.append(pd.concat(sample_result, ignore_index=True))
    result_pd = pd.concat(result, ignore_index=True)
    result_pd.to_csv("NF2_interAB_C.csv", index=False)

#%%
if __name__ == "__main__":
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    #type2_interABC(sparql)
    type2_interAB_C(sparql)
 # %%
