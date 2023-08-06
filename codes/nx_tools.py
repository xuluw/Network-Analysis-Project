import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil
import os
import imageio.v2 as imageio
import statsmodels.api as sm


#------------------------------
# ANIMATION PLOT
#------------------------------
def animate(networks,pos_type="circular",path="output.gif",plot_every=1):

    # BUILD A TEMPORARY FOLDER TO STORE tmp-i.png
    def build_tmp_folder(out_dir_path):
        if(os.path.exists(out_dir_path)):
            shutil.rmtree(out_dir_path)
        os.mkdir(out_dir_path)

    tmp_folder = "tmp_png"
    build_tmp_folder(tmp_folder)

    
    for network in networks:
         
         # DECICE THE NETWORK LAYOUT
         if pos_type=="circular":
              pos = nx.circular_layout(network)
         elif pos_type=="random":
              pos = nx.random_layout(network)
         elif pos_type=="spring":
              pos = nx.spring_layout(network)
         

         # If it's a directed graph
         if nx.is_directed(network):
              # node size = in-degree
              in_degree = list(dict(network.in_degree).values())
              node_sizes = [x*500+150 for x in in_degree]
              # node color = out-degree
              cmap=plt.cm.get_cmap('summer')
              out_degree = list(dict(network.out_degree).values())
              node_colors = [cmap(u/(0.01+max(out_degree))) for u in out_degree]

                   
              #INITIALIZE FIGURE AND PLOT
              fig, ax = plt.subplots()
              fig.set_size_inches(5, 5)
              
              #GET MIN AND MAX POSITION
              tmpx=[]; tmpy=[]
              for i in pos.keys():
                   tmpx.append(pos[i][0])
                   tmpy.append(pos[i][1])
              Lxmin=min(tmpx)-0.2; Lxmax=max(tmpx)+0.2
              Lymin=min(tmpy)-0.2; Lymax=max(tmpy)+0.2
              
              #DRAW BOX
              ax.axhline(y=Lymin)
              ax.axvline(x=Lxmin)
              ax.axhline(y=Lymax)
              ax.axvline(x=Lxmax)

              # PLOT NETWORK
              nx.draw(network,
                      with_labels=True,
                      edgecolors="white",
                      node_color=node_colors,
                      node_size=node_sizes,
                      font_color='black',
                      font_size=14,
                      pos=pos,
                      ax=ax
                      )
              plt.savefig(tmp_folder+"/tmp-"+str(plot_every)+".png")
              plt.close()   # Don't show the graph in ".ipynb"
              plot_every+=1

         else:
              # node size = node color = degree
              degree = list(dict(network.degree).values())
              # node size
              node_sizes = [x*300 for x in degree]
              # node color
              cmap=plt.cm.get_cmap('summer')
              node_colors = [cmap(u/(0.01+max(degree))) for u in degree]

              #INITIALIZE FIGURE AND PLOT
              fig, ax = plt.subplots()
              fig.set_size_inches(6, 6)
              
              #GET MIN AND MAX POSITION
              tmpx=[]; tmpy=[]
              for i in pos.keys():
                   tmpx.append(pos[i][0])
                   tmpy.append(pos[i][1])
              Lxmin=min(tmpx)-0.2; Lxmax=max(tmpx)+0.2
              Lymin=min(tmpy)-0.2; Lymax=max(tmpy)+0.2
              
              #DRAW BOX
              ax.axhline(y=Lymin)
              ax.axvline(x=Lxmin)
              ax.axhline(y=Lymax)
              ax.axvline(x=Lxmax)

              # PLOT NETWORK
              nx.draw(network,
                      with_labels=True,
                      edgecolors="white",
                      node_color=node_colors,
                      node_size=node_sizes,
                      font_color='black',
                      font_size=14,
                      pos=pos,
                      ax=ax
                      )
              plt.savefig(tmp_folder+"/tmp-"+str(plot_every)+".png")
              plt.close() # Don't show the graph in ".ipynb"
              plot_every+=1

    
    file_names = os.listdir(tmp_folder)
    # Sort the file names based on the numbers present in the filename
    sorted_file_names = sorted(file_names, key=lambda x: int(''.join(filter(str.isdigit, x))))
    # Create the GIF
    images = list(map(lambda filename: imageio.imread(tmp_folder+"/"+filename), sorted_file_names))
    imageio.mimsave(path, images, duration = 400)#/len(file_names) ) # modify the frame duration as needed

    # Delete the temporary folder and its contents
    shutil.rmtree(tmp_folder)




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
# AVERAGE DEGREE
#------------------------------
def ave_degree(G):

    # If it's a directed graph
    if nx.is_directed(G):
        avg_in_degree = np.mean(list(dict(G.in_degree()).values()))
        avg_out_degree = np.mean(list(dict(G.out_degree()).values()))

        print("The average in-degree is", avg_in_degree)
        print("The average out-degree is", avg_out_degree)

    # If it's an un-directed graph
    else:
         avg_degree=np.mean(list(dict(G.degree()).values()))

         print("The average degree is", avg_degree)




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

    BINS=40
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
# NETWORK PLOTTING FUNCTION
#------------------------------
def plot_network(G,node_color="degree",layout="random"):
    
    # POSITIONS LAYOUT
    N=len(G.nodes)
    if(layout=="spring"):
        # pos=nx.spring_layout(G,k=50*1./np.sqrt(N),iterations=100)
        pos=nx.spring_layout(G)

    if(layout=="random"):
        pos=nx.random_layout(G)

    #INITALIZE PLOT
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    # NODE COLORS
    cmap=plt.cm.get_cmap('Greens')

    # DEGREE 
    if node_color=="degree":
            centrality=list(dict(nx.degree(G)).values())
  
    # BETWENNESS 
    if node_color=="betweeness":
            centrality=list(dict(nx.betweenness_centrality(G)).values())
  
    # CLOSENESS
    if node_color=="closeness":
            centrality=list(dict(nx.closeness_centrality(G)).values())

    # NODE SIZE CAN COLOR
    node_colors = [cmap(u/(0.01+max(centrality))) for u in centrality]
    node_sizes = [4000*u/(0.01+max(centrality)) for u in centrality]

    # #PLOT NETWORK
    nx.draw(G,
            with_labels=True,
            edgecolors="black",
            node_color=node_colors,
            node_size=node_sizes,
            font_color='white',
            font_size=18,
            pos=pos
            )

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
        print("	          ",list(sort_dict.values())[0:12])
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
            print("CLOSENESS CENTRALITY")
            centrality_stats(nx.closeness_centrality(G))
            print("BETWEEN CENTRALITY")
            centrality_stats(nx.betweenness_centrality(G))
            print("EIGENVECTOR CENTRALITY")
            centrality_stats(nx.eigenvector_centrality(G))

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

