import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm



#------------------------------
# NETWORK CENTRALITY CORRELATION PLOTS
#------------------------------
def plot_centrality_correlation(G,path=""):

    # If it's a directed graph
    if nx.is_directed(G):
         
         # In-degree
         in_degree = nx.in_degree_centrality(G).values()

         # Out-degree
         out_degree = nx.out_degree_centrality(G).values()

         # In-closeness
         in_closeness = nx.closeness_centrality(G).values()

         # Out-closeness
         out_closeness = nx.closeness_centrality(G.reverse()).values()

         # Betweenness
         betweenness =nx.betweenness_centrality(G.to_undirected()).values()

         df = pd.DataFrame({"In-degree":in_degree, 
                            "Out-degree":out_degree, 
                            "In-closeness":in_closeness, 
                            "Out-closeness":out_closeness, 
                            "betweenness":betweenness})
         g = sns.pairplot(df, diag_kind='hist')

    # If it's an un-directed graph
    else:
         # Degree
         degree = nx.degree_centrality(G).values()
         
         # Closeness
         closeness = nx.closeness_centrality(G).values()
         
         # Betweenness
         betweenness = nx.betweenness_centrality(G).values()
         
         df = pd.DataFrame({"Degree":degree, 
                            "Closeness":closeness, 
                            "Betweenness":betweenness})
         g = sns.pairplot(df, diag_kind='hist')

    # Save and plot
    if path!="":
        plt.savefig(path, dpi=300, format='pdf')
    plt.show()




#------------------------------
# PLOT DEGREE DISTRIBUTION
#------------------------------
def plot_degree_distribution(G,type="in",path=""):

    from scipy.optimize import curve_fit

    # If it's an un-directed graph
    if not nx.is_directed(G):
         degree = list(dict(G.degree).values())
    # If it's a directed graph
    else:
         if type=="in":
              degree = list(dict(G.in_degree).values())
         if type=="out":
              degree = list(dict(G.out_degree).values())

    BINS=35
    N=nx.number_of_nodes(G)
    FS=14

    #General plot
    fig, axs = plt.subplots(1,4)
    fig.set_size_inches(28, 7)

    #1st column: PDF
    sns.histplot(degree, bins=BINS, stat="density", kde=False, ax=axs[0])
    axs[0].set_xlabel("Degree", fontsize=FS)
    axs[0].set_ylabel("Probability", fontsize=FS)

    #2nd column: PDF on log-log scale
    counts1, bins1=np.histogram(degree, bins=BINS, density=False)
    bins1=(np.array(bins1[1:])+np.array(bins1[0:-1]))/2.0

    axs[1].plot(bins1, counts1/N, "o-")#,color="orange")
    axs[1].set_xlabel("Degree (log)", fontsize=FS)
    axs[1].set_ylabel("Probability (log)", fontsize=FS)
    axs[1].set_xscale('log'); axs[0].set_yscale('log')

    #3rd column: cCDF
    sns.ecdfplot(data=degree, complementary=True, ax=axs[2], marker='o')
    axs[2].set_xlabel("Degree", fontsize=FS)
    axs[2].set_ylabel("cCDF", fontsize=FS)

    #4th column: cCDF on log-log scale
    sns.ecdfplot(data=degree, complementary=True, ax=axs[3])#, color="orange")
    axs[3].set_xlabel("Degree (log)", fontsize=FS)
    axs[3].set_ylabel("cCDF (log)", fontsize=FS)
    axs[3].set_xscale('log'); axs[3].set_yscale('log')

    if path!="":
         plt.savefig(path)
    plt.show()





#------------------------------
# NETWORK SUMMARY FUNCTION
#------------------------------
def network_summary(G):

    def centrality_stats(x):
        x1=dict(x)
        x2=np.array(list(x1.values())); #print(x2)
        print("	min:" ,min(x2))
        print("	mean:" ,np.mean(x2))
        print("	median:" ,np.median(x2))
        # print("	mode:" ,stats.mode(x2)[0][0])
        print("	max:" ,max(x2))
        x=dict(x)
        sort_dict=dict(sorted(x1.items(), key=lambda item: item[1],reverse=True))
        print("	top nodes:",list(sort_dict)[0:12])
        print("	          ",list(sort_dict.values())[0:10])
        print("	tail nodes:",list(sort_dict)[-10:])
        print("	          ",list(sort_dict.values())[-10:])

    try: 
        print("GENERAL")
        print("	number of nodes:",len(list(G.nodes)))
        print("	number of edges:",len(list(G.edges)))
        
        print("	is_directed:", nx.is_directed(G))
        print("	is_weighted:" ,nx.is_weighted(G))

        if(nx.is_directed(G)):
            print(" is_strongly_connected:",nx.is_strongly_connected(G))
            print(" is_weakly_connected:",nx.is_weakly_connected(G))
            print("IN-DEGREE (NORMALIZED)")
            centrality_stats(nx.in_degree_centrality(G))
            print("OUT-DEGREE (NORMALIZED)")
            centrality_stats(nx.out_degree_centrality(G))
            print("DENSITY:" ,nx.density(G))
            print("AVERAGE CLUSTERING COEFFICIENT: ", nx.average_clustering(G))
            print("DEGREE ASSORTATIVITY COEFFICIENT: ", nx.degree_assortativity_coefficient(G))
            #CENTRALITY 
            print("DEGREE (NORMALIZED)")
            centrality_stats(nx.degree_centrality(G))
            print("CLOSENESS CENTRALITY (inward)")
            centrality_stats(nx.closeness_centrality(G))
            print("CLOSENESS CENTRALITY (outward)")
            centrality_stats(nx.closeness_centrality(G.reverse()))
            print("BETWEEN CENTRALITY")
            centrality_stats(nx.betweenness_centrality(G,weight="weight"))
            print("EIGENVECTOR CENTRALITY")
            centrality_stats(nx.eigenvector_centrality(G,weight="weight"))

            if(nx.is_strongly_connected(G)):
                print("DIAMETER:" ,nx.diameter(G))
                print("RADIUS:" ,nx.radius(G))
                print("AVERAGE SHORTEST PATH LENGTH: ", nx.average_shortest_path_length(G))

        else:
            print("	number_connected_components", nx.number_connected_components(G))
            print("	number of triangle: ",len(nx.triangles(G).keys()))
            print("	density:" ,nx.density(G))
            print("	average_clustering coefficient: ", nx.average_clustering(G))
            print("	degree_assortativity_coefficient: ", nx.degree_assortativity_coefficient(G))
            print("	is_tree:" ,nx.is_tree(G))

            if(nx.is_connected(G)):
                print("	diameter:" ,nx.diameter(G))
                print("	radius:" ,nx.radius(G))
                print("	average_shortest_path_length: ", nx.average_shortest_path_length(G))

            #CENTRALITY 
            print("DEGREE (NORMALIZED)")
            centrality_stats(nx.degree_centrality(G))

            print("CLOSENESS CENTRALITY")
            centrality_stats(nx.closeness_centrality(G))

            print("BETWEEN CENTRALITY")
            centrality_stats(nx.betweenness_centrality(G))

            print("EIGENVECTOR CENTRALITY")
            centrality_stats(nx.eigenvector_centrality(G))

    except:
        print("unable to run")




#------------------------------
# ISOLATE GCC
#------------------------------
def isolate_GCC(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True) 
    nodes_in_giant_comp = comps[0]
    return nx.subgraph(G, nodes_in_giant_comp)

