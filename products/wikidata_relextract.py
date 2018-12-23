import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
import numpy as np
import json
from scipy.sparse import csr_matrix
import networkx as nx
from collections import defaultdict
import os



def make_edge_set(): return ([],([],[]))
def add_edge(e, i,j):
    (v,(row,col)) = e
    row.append(i)
    col.append(j)
    v.append(1)




# Build dicts based on properties in Wikidata. 

Rel_toPIDs ={'airline_hub':'P113', 'lyrics_by':'P676', 'place_of_publication':'P291'}


numb_success = 0
failed_requests = []
json_errors = []
empty_results = []
dense_rels = []

for key, val in Rel_toPIDs.items():
    curr_query = '''PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?item ?instance_of WHERE {
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        OPTIONAL { ?item wdt:%s ?instance_of. }
    }
    '''%(val)

    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    curr_data = requests.get(url, params={'query': curr_query, 'format': 'json'})
    print(curr_data.status_code)

    if curr_data.status_code != 200:
        print("Failed rel from HTTPS:"+str(key))
        failed_requests.append(key)

    else:
        try:
            curr_data = curr_data.json(strict=False)       
            # Write the edgelists and dicts for the rel. 

            rel = key
            QtoIDs = dict()
            IDtoQs = dict()
            e = make_edge_set()
            counter = 0
            triple_count = 0

            if 'instance_of' in (curr_data['results']['bindings'][0]):
                for triple in curr_data['results']['bindings']:
                    instance_of = triple['instance_of']['value'].split("/")[-1]
                    item = triple['item']['value'].split("/")[-1]
                    if item not in QtoIDs.keys():
                        QtoIDs[item] = counter
                        IDtoQs[counter] = item
                        counter+=1
                    if instance_of not in QtoIDs.keys():
                        QtoIDs[instance_of] = counter
                        IDtoQs[counter] = instance_of
                    add_edge(e, QtoIDs[item], QtoIDs[instance_of])
                    add_edge(e, QtoIDs[instance_of], QtoIDs[item])
                    triple_count+=1


                # Take the largest connected component for the relationship.

                n = len(QtoIDs)
                X = csr_matrix(e, shape=(n, n))
                G = nx.from_scipy_sparse_matrix(X)
                Gc = max(nx.connected_component_subgraphs(G), key=len)
                print(rel)
                print("Total number of unique entities: "+str(G.number_of_nodes()))
                print("Total number of nodes in lcc: "+str(Gc.number_of_nodes()))
                Gc_final = nx.convert_node_labels_to_integers(Gc, ordering="decreasing degree", label_attribute="old_label")

                if (Gc.number_of_edges()>100*Gc.number_of_nodes()):
                    dense_rels.append(key)

                #Create the dict for old-id <-> new-id matching for QIDs.
                RefDict = Gc_final.node
                IDtoQs_f = dict()
                QtoIDs_f = dict()
                for new_idx in RefDict.keys():
                    old_idx = RefDict[new_idx]['old_label']
                    curr_Q = IDtoQs[old_idx]
                    IDtoQs_f[new_idx] = curr_Q
                    QtoIDs_f[curr_Q] = new_idx

                
                #Write the final edgelist and dump IDstoQs_f dict.
                nx.write_edgelist(Gc_final, "data/wikidata_edges/"+str(rel)+"_lcc.edges",data=False)
                json.dump(IDtoQs_f, open("data/wikidata_edges/"+str(rel)+"_IDstoQs.txt","w"))

            else:
                empty_results.append(key)

        except json.decoder.JSONDecodeError:
            json_errors.append(key)

     


print("Failed HTTP requests:")
print(failed_requests)
print("JSONDecodeErrors")
print(json_errors)
print("Empty rels")
print(empty_results)
print("Dense rels")
print(dense_rels)
    
