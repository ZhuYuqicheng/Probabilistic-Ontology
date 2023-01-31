#%%
from SPARQLWrapper import SPARQLWrapper, JSON
from itertools import combinations
import pandas as pd

def single_class_result(sparql, cls_str):
    # count A and B
    subcls_query = """SELECT DISTINCT ?subclass (COUNT(?instance) AS ?number)"""
    subcls_query += f" WHERE{{?subclass rdfs:subClassOf* <{cls_str}> . "
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
    print((cls_str, x["subclass"]["value"], numerator/denominator))
    return pd.DataFrame(result, columns=["ConceptA", "ConceptB", "Probability"])
def get_subsumption(sparql):
    list_query = """SELECT DISTINCT ?subclass2 ?subclass1"""
    list_query += f" WHERE{{?subclass1 rdfs:subClassOf* <http://dbpedia.org/ontology/Person> . "
    list_query += f"?subclass2 rdfs:subClassOf* ?subclass1 . "
    list_query += f"FILTER (?subclass1 != ?subclass2)}}"
    sparql.setQuery(list_query)
    list_out = sparql.queryAndConvert()
    result = []
    for x in list_out["results"]["bindings"]:
        result.append((x["subclass2"]["value"], x["subclass1"]["value"], 1.0))
    return pd.DataFrame(result, columns=["ConceptA", "ConceptB", "Probability"])
def get_domain_result(sparql, domain):
    list_query = """SELECT DISTINCT ?subclass"""
    list_query += f" WHERE{{?subclass rdfs:subClassOf* <{domain}>}}"
    sparql.setQuery(list_query)
    list_out = sparql.queryAndConvert()
    cls_list = [x["subclass"]["value"] for x in list_out["results"]["bindings"]]
    buffer = []
    for cls_str in cls_list:
        buffer.append(single_class_result(sparql, cls_str))
    return pd.concat(buffer, ignore_index=True)

if __name__ == "__main__":
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    domains = ["http://dbpedia.org/ontology/Person",
            "http://dbpedia.org/ontology/Organisation",
            "http://dbpedia.org/ontology/Place",
            "http://dbpedia.org/ontology/Species",
            "http://dbpedia.org/ontology/Event",
            "http://dbpedia.org/ontology/Work",
            "http://dbpedia.org/ontology/Activity",
            "http://dbpedia.org/ontology/Food"]
    type1_axioms = []
    for domain in domains:
        type1_axioms.append(get_domain_result(sparql, domain))
    type1_axioms_result = pd.concat(type1_axioms, ignore_index=True)
    type1_axioms_result.to_csv("NF1.csv", index=False)