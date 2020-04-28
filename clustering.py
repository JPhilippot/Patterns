
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os


################## PARAMETERS ###############
# DIRECTORY STANDS FOR THE DIRECTORY TO READ
DIRECTORY="./Kmeans"
NB_CLUSTERS=4
JSONFILE="myJson.json"
# CLASS_TEST IS THE NUMBER OF THE CLASS TO TEST
# IT MUST BE SET AT THE HIGHEST RANK
CLASS_TEST=3
# TEST_EXIST allows to know if there is a test to perform
# it is upadated if there is a directory with the keyword test for testing a class
TEST_EXIST=False

#############################################

#############################################
#########       FUNCTIONS      ##############
############################################# 
def get_clusters_forclasses(y,y_predict):
    df_y_nbdifferentvalues=pd.DataFrame(y)
    list_unique_value_y=df_y_nbdifferentvalues[0].unique()
    clusters=[]
    for i in range (len(y_predict)):
        dict_clusters={}
        for j in range (len(list_unique_value_y)):
            if y[i]==list_unique_value_y[j]:
                dict_clusters['y']=y[i]
                dict_clusters['y_predict']=y_predict[i]
                clusters.append(dict_clusters)
    df_clusters = pd.DataFrame(clusters)
    clustersperclass=[]
    objectsperclusterperclass=[]
    for j in range (len(list_unique_value_y)):
        objectsperclusterperclass=[]
        dict_clustersperclass={}
        df_y=df_clusters.loc[df_clusters['y'] == list_unique_value_y[j]]
        print ("Clusters de la classe #",
               list_unique_value_y[j],":",df_y['y_predict'].unique())
        dict_clustersperclass['class']=list_unique_value_y[j]
        dict_clustersperclass['clusters']=df_y['y_predict'].unique()
        dict_clustersperclass['nbobjects']=np.nan
        df=pd.DataFrame(df_y['y_predict'].value_counts())
        df = df.reset_index()
        df.columns = ['cluster', 'counts']
        for nb_objects in range(len(df)):
            val_to_test=df['cluster'].loc[nb_objects] #number of the cluster
            for nb_clusters in range (len(dict_clustersperclass['clusters'])):
                if val_to_test==dict_clustersperclass['clusters'][nb_clusters]:
                        objectsperclusterperclass.append(df['counts'].loc[nb_objects])    
        dict_clustersperclass['nbobjects']=objectsperclusterperclass            
        clustersperclass.append(dict_clustersperclass)
        
    df_clustersperclass = pd.DataFrame(clustersperclass)
    return df_clusters,df_clustersperclass


def build_dataframe_forJson(df_objectsclasslayers_ext):
    '''
    This function creates a dataframe that will be used to generate the links and nodes 
    of the Json for the Sankey.
    
    Input : df_objectsclasslayers_ext example
                class  l0  l1  l2    l0r    l1r    l2r  predict
            0      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
            1      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
            2      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
            3      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
            4      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
            
    Ouput : 
        - a dataframe for the links (df_forJsonLinks)  
        - a dataframe for the nodes (df_forJsonNodes)
    Example of df_forJsonLinks:
                    class source target  value
            0       0     X0    Cl1_1     50
            1       1     X1    Cl1_0     25
            2       1     X1    Cl1_2     10
            3       1     X1    Cl1_3     15
            4       3     X3    Cl1_1     50
            5       0  Cl1_1    Cl2_1     50
    Each line stands for an entry with the class, source, target and value columns.
    At line 0, 0 stands for the number of the initial, 
    X0 means the input of the neural network. Each class has the number of the corresponding layer.
    For instance Cl1_1 means that it is a the cluster C1 at the layer 1 (l1).
    Value stands for the number of objects  
    
    Example of df_forJsonNodes:
                source class shared
            0      X0     0  false
            1      X1     1   true
            2      X3     3  false
            3   Cl1_1     0   true
            4   Cl1_0     1   true
    Each line stands for a node. Source is the name, class is the initial class of the node.
    if shared is true that means that other clusters are using this cluster (cluster multiclass or
    test class)
        
    '''
    #  build the dataframe to create the Json file
    # Get the list of the columns, this will be used to combine the layers two per two
    # The dataframe has n columns, 2 for class and predict then (n-2)/2 is the number
    # where the name for Json is available, i.e. X0
    start_column=int((df_objectsclasslayers_ext.shape[1])/2)
    list_columns=list(df_objectsclasslayers_ext)
    del list_columns[0:start_column]
    df_forJsonLinks=pd.DataFrame()#the dataframe that will store the result
    # Link part
    # the following loop will construct the dataframe from the input to the last layer
    for i in range(len(list_columns)-1):
        df_val=df_objectsclasslayers_ext[['class',list_columns[i],list_columns[i+1]]]
        result=df_val.groupby(['class',list_columns[i],list_columns[i+1]]).size().reset_index()
        result.columns = ['class','source', 'target','value']
        df_forJsonLinks=pd.concat([df_forJsonLinks,result])    
    df_forJsonLinks.index = range(len(df_forJsonLinks.index))
    
    # Node part
    df_forJsonNodes=pd.DataFrame()
    df_forJsonNodes=df_forJsonLinks[['source','class']].copy()
    df_forJsonNodes['shared']='false'
    df_dup=pd.concat(g for _, g in df_forJsonNodes.groupby([df_forJsonNodes['source']]) if len(g) > 1)
    df_dup=df_dup.reset_index()
    for nb_transform in range (len(df_dup)):
        df_forJsonNodes['shared'].iloc[df_dup['index'][nb_transform]]='true'
    df_forJsonNodes=df_forJsonNodes.drop_duplicates(subset ='source').reset_index(drop=True)
    # adding the number of classes from predict, i.e. the last layer
    df_nb = df_objectsclasslayers_ext['predict'].value_counts().rename_axis('predict').reset_index(name='counts')
    nb=df_nb['predict']
    for nb_classes in range (len(nb)):
        df_forJsonNodes = df_forJsonNodes.append({'source' : str(nb[nb_classes]) , 
                                      'class' : str(nb[nb_classes]),
                                      'shared': 'false'},
                                        ignore_index=True)
    df_forJsonNodes=df_forJsonNodes.reset_index(drop=True)  
    
    return df_forJsonLinks,df_forJsonNodes



###############################################
########             SEARCH FILES         #####
###############################################

# Extract the list of directories
print ("Initialisation part: reading the files in the directory",DIRECTORY)
content_directory=os.listdir(DIRECTORY)
list_dir=[]
for file_dir in range(len(content_directory)):
    if os.path.isdir(DIRECTORY+'/'+content_directory[file_dir]):
        list_dir.append(DIRECTORY+'/'+content_directory[file_dir]) 


# Get the different files in order to apply the clustering
list_files_to_learn=[]
list_files_to_test=[]
for nb_files in range (len(list_dir)):
    content=os.listdir(list_dir[nb_files])
    if "test" not in list_dir[nb_files]:
        for nb_files_in_directory in range (len(content)):
            if "DS" not in content[nb_files_in_directory]:
                list_files_to_learn.append(list_dir[nb_files]+'/'+content[nb_files_in_directory])
    else:
        TEST_EXIST=True
        for nb_files_in_directory in range (len(content)):
            if "DS" not in content[nb_files_in_directory]:
                list_files_to_test.append(list_dir[nb_files]+'/'+content[nb_files_in_directory])
                
# sort by alphabetical order the files                
list_files_to_learn=sorted(list_files_to_learn)     
if TEST_EXIST:
    list_files_to_test=sorted(list_files_to_test)   
     

###############################################
##################   CLUSTERING ###############
###############################################

print ("Compute the clustering for the original data")
'''
Apply the kmeans for all layer for the original data

- k_means_perlayer stores the return of the k_means for each layer. It will be used
for predicting the test
- cluster_classes_perlayer is an array of dataframe. It has as many dataframe as layers.
Each dataframe is obtained by the get_clusters_forclasses function
Example:
    [class clusters nbobjects
0    0.0   [1, 3]  [28, 22]
1    1.0   [2, 0]  [38, 12],    
    class clusters nbobjects
0    0.0   [3, 1]  [26, 24]
1    1.0   [0, 2]  [36, 14]]
Each dataframe has for each layer the class, the corresponding clusters as well as the number of objects
for each cluster. For instance,  0.0   [1, 3]  [28, 22]  means that at the first layer, for the class 0, 
there are 2 clusters (1 and 3) and the number of objects of cluster 1 is 28 (resp. 22 for cluster 3).
- objects_cluster_perlayer is an array of dataframe which has as many dataframes as layers.
Each dataframe is obtained by the get_clusters_forclasses function. For each layer it stores for each object 
the class and the y_predict from the k-means, i.e. the number of the cluster.
Example:
    [      y  y_predict
    0   0.0          1
    1   0.0          1
    2   0.0          1
    ...],
    [      y  y_predict
    0   0.0          2
    1   0.0          2
0   0.0          2 means that at the second layer (2nd row of the array), for the object 0 (index), 
the class (y) is 0 and the predicted cluster is 2.
'''
kmeans_per_layer=[]
cluster_classes_perlayer=[]
objects_cluster_perlayer=[]
for nb_layers in range (len(list_files_to_learn)):
    print("Layer",nb_layers+1)
    df = pd.read_csv(list_files_to_learn[nb_layers], sep = ',', header = None)
    array = df.values
    y = array[:,0]
    X = array[:,1:df.shape[1]] 
    kmeans=KMeans(n_clusters=NB_CLUSTERS, random_state=30).fit(X)
    y_predict = kmeans.predict(X)
    title="iris_l1_8"
    #plot_clusters_2D ("irisl1.png",legend,title,X,y,y_predict1, nb_clusters)
    df_clusters,df_clustersperclass=get_clusters_forclasses(y,y_predict)
    cluster_classes_perlayer.append(df_clustersperclass)
    objects_cluster_perlayer.append(df_clusters)
    kmeans_per_layer.append(kmeans)

if TEST_EXIST:
    print ("\nCompute the clustering for the test class\n")
    # Apply the kmeans predict for all layer for the test data
    cluster_test_perlayer=[]
    objectstest_cluster_perlayer=[]
    if list_files_to_test:
        for nb_layers in range (len(list_files_to_test)):
            print("Layer",nb_layers+1)
            df = pd.read_csv(list_files_to_test[nb_layers], sep = ',', header = None)
            array = df.values
            y = array[:,0]
            X = array[:,1:df.shape[1]] 
            y_predict = kmeans_per_layer[nb_layers].predict(X)
            title="iris_l1_8"
            #plot_clusters_2D ("irisl1.png",legend,title,X,y,y_predict1, nb_clusters)
            df_clusters_test,df_clustersperclass=get_clusters_forclasses(y,y_predict)
            cluster_test_perlayer.append(df_clustersperclass)
            objectstest_cluster_perlayer.append(df_clusters_test)
            kmeans_per_layer.append(kmeans)




###############################################
########  CREATION OF JSON FILE ###############
###############################################

# Build the array of objects, class, layer1, layer2, ..., layern
'''
objects_cluster is an array of array (i.e. the number of objects in the dataset
 where for each object we have :
   - its class, the cluster of layer1, the cluster of layer 2, ...
Example :  
    [[0. 1. 3. 1.]
     [0. 1. 3. 1.]
     [0. 1. 3. 1.]
The first three objects of the dataset are in class 0 (0.), are in cluster 1 at layer 0, cluster 2 
at layer 1, 1 at layer 2.
Be carefull layers are numeroted from 0.      
'''
objects_cluster=objects_cluster_perlayer[0]['y'].copy()
for nb_layers in range (len(objects_cluster_perlayer)):
    objects_cluster=np.c_[objects_cluster,objects_cluster_perlayer[nb_layers]['y_predict']]

if TEST_EXIST:
    # Build the array for test objects, class, layer1, layer2, ..., layern
    objectstest_cluster=objectstest_cluster_perlayer[0]['y'].copy()
    for nb_layers in range (len(objects_cluster_perlayer)):
        objectstest_cluster=np.c_[objectstest_cluster,objectstest_cluster_perlayer[nb_layers]['y_predict']]
            

'''
Creation of a unique dataframe for original data and test
At the end,         
df_object_class_layers is the dataframe that store for each occurrence 
its class and the clusters for each layer
Example:
   class   l0   l1   l2
0    0.0  1.0  3.0  1.0
1    0.0  1.0  3.0  1.0
2    0.0  1.0  3.0  1.0
3    0.0  1.0  3.0  1.0
4    0.0  1.0  3.0  1.0
'''
nb_layers= len(objects_cluster_perlayer)   
columname=[]
columname.append('class')
for i in range (nb_layers):
    columname.append('l'+str(i+1))
df_original=pd.DataFrame(objects_cluster,columns=columname)
# nb_X_original will be used later to know how many test are added
NB_X_ORIGINAL=df_original.shape[0]
if TEST_EXIST:
    df_test=pd.DataFrame(objectstest_cluster, columns=columname)
    # Concatenate original and test
    df_object_class_layers=pd.concat([df_original, df_test])
    df_object_class_layers = df_object_class_layers.reset_index(drop=True)
else:
    df_object_class_layers=df_original.copy()
    df_object_class_layers = df_object_class_layers.reset_index(drop=True)


############ CREATE AN EXTENDED DATAFRAME ##############
'''
Creation of a unique dataframe for original data and test. It is an extension
of the previous one :
    - convert in int
    - add columns with the name of the clusters for Json (l0r, l1r, ...)
    - add a new column (predict) for the prediction of the value
    - the lines corresponding to the test data are replaced by the number of the class, i.e.
    if we have original data as class 0 and 1, the test class is 2.
df_objectsclasslayers_ext example:
    class  l0  l1  l2    l0r    l1r    l2r  predict
0      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
1      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
2      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
3      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
4      0   1   1   1  Cl1_1  Cl2_1  Cl3_1        0
'''
df_objectsclasslayers_ext=df_object_class_layers.copy()
# convert to int
df_objectsclasslayers_ext=df_objectsclasslayers_ext.astype(int)

df_tempo=df_objectsclasslayers_ext['class'].copy()
if TEST_EXIST:
    df_objectsclasslayers_ext['class'][NB_X_ORIGINAL:]=CLASS_TEST
for nb_layers in range(df_objectsclasslayers_ext.shape[1]-1):
    if nb_layers==0:
        df_objectsclasslayers_ext['l0r']=df_objectsclasslayers_ext['class'].apply(lambda x: 'X'+str(int(x)))
    else:
        df_objectsclasslayers_ext['l'+str(nb_layers)+'r']=\
        df_objectsclasslayers_ext['l'+str(nb_layers)].apply(lambda x: 'Cl'+str(nb_layers)+'_'+str(int(x)))
df_objectsclasslayers_ext['predict']=df_tempo
df_objectsclasslayers_ext['class']=df_objectsclasslayers_ext['class'].apply(lambda x: 'C'+str(x))
df_objectsclasslayers_ext['predict']=df_objectsclasslayers_ext['predict'].apply(lambda x: 'C'+str(x))
print (df_objectsclasslayers_ext['class'].head())


# CREATE THE JSON

# Call the function to create the dataframe for the Json
df_forJsonLinks,df_forJsonNodes=build_dataframe_forJson(df_objectsclasslayers_ext)    

# Create the link part of the Json
links_json="\"links\":[\n"
for i in range (len(df_forJsonLinks)):
    links_json+="{"+"\"source\""+':'+"\""+str(df_forJsonLinks['source'].loc[i])+"\","
    links_json+="\"target\""+':'+"\""+str(df_forJsonLinks['target'].loc[i])+"\","
    links_json+="\"value\""+':'+"\""+str(df_forJsonLinks['value'].loc[i])+"\","
    links_json+="\"classname\""+':'+"\""+str(df_forJsonLinks['class'].loc[i])+"\"}"
    if i != len(df_forJsonLinks)-1:
        links_json+=",\n"
    else:
        links_json+="\n]"# pas de saut de ligne pour mettre la virgule avant node 


# Create the node part of the Json
nodes_json="\"nodes\":[\n"
for i in range (len(df_forJsonNodes)):
    nodes_json+="{"+"\"name\""+':'+"\""+str(df_forJsonNodes['source'].loc[i])+"\","
    nodes_json+="\"classname\""+':'+"\""+str(df_forJsonNodes['class'].loc[i])+"\","
    nodes_json+="\"shared\""+':'+"\""+str(df_forJsonNodes['shared'].loc[i])+"\"}"
    if i != len(df_forJsonNodes)-1:
        nodes_json+=",\n"
    else:
        nodes_json+="\n]\n"  

final_json='{\n'+links_json+',\n'+nodes_json+'}'
print ("The final json is \n",final_json)

# save the file
with open(JSONFILE, 'w') as f:
    f.write(final_json)
