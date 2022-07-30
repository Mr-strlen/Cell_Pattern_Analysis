## 调用global features from vaa3d plugin in获取结果
import subprocess
import re,os,csv,math
import numpy as np
import shutil
from sklearn.decomposition import PCA
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns 
import networkx as nx

global path
path=r'.\Data'
global vaa3d_path
with open('vaa3d.conf') as f:
    t=str(f.readline())
    f.close()
vaa3d_path=t
global package_list
package_list=['numpy','scikit-learn','matplotlib','seaborn','networkx','multiprocess']

def install_package(package_list):
    p = os.popen("pip list --format=columns")  # 获取所有包名 直接用 pip list 也可获取
    pip_list = p.read()  # 读取所有内容
    for package_name in package_list:
      package_name = package_name.replace("_", "-")  # 下载pip fake_useragent 包时  包名是:fake-useragent
      if package_name in pip_list:
          print("{} installed".format(package_name))
      else:
          print("Not install {}! Automatically installed soon. Please wait".format(package_name))
          p = os.popen("pip install {}".format(package_name))
          if "Success" in p.read():
              print("Install {} success!".format(package_name))
def Init():
    install_package(package_list)

def GetFeature(filename):
    readpath=path+'\\'+filename
    outpath='.\\Feature\\'+filename.split('.')[0]+'_feature.txt'
    
    cmd=vaa3d_path+r'\vaa3d_msvc.exe /x '+vaa3d_path+r'\plugins\neuron_utilities\global_neuron_feature\global_neuron_feature.dll /f compute_feature /i '+readpath
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # process.wait()
    command_output = process.stdout.read().decode()
    
    c1=re.search('compute Feature', command_output)
    c2=re.search('the plugin preprocessing takes', command_output)
    if c1 is None or c2 is None:
        print(filename+' no result')
        return
    res=command_output[c1.span()[1]+4:c2.span()[0]-6]
    with open(outpath,"w+",newline='') as f:
        f.write(res)
        f.close()

def run__pool():  # main process
    from multiprocessing import Pool
    cpu_worker_num = 8
    global filenames
    for root,dirs,files in os.walk(path,topdown=True):
        filenames=files
    if os.path.exists('./Feature'):
        shutil.rmtree('./Feature')
    os.mkdir('./Feature')
    with Pool(cpu_worker_num) as p:
        p.map(GetFeature, filenames)
    
def GetFileFeature(contents):
    temp=[]
    for x in contents:
        t=x.replace('\t','')
        t=t.replace('\n','')
        t=t.split(':')
        temp.append(float(t[1]))
    return temp

def GetFeatureCSV():
    feature_map=[]
    for root,dirs,files in os.walk('./Feature',topdown=True):
        for file in files:
            temp=[file]
            with open('.\\Feature\\'+file) as file_object:
                contents = file_object.readlines()
                file_object.close()
            temp.extend(GetFileFeature(contents))
            feature_map.append(temp)

    with open("Feature.csv","w+",newline='') as f:
        csv_writer = csv.writer(f)
        t=['id','N_node','Soma_surface','N_stem','f0','f1','f2',\
            'Number of Nodes','Soma Surface','Number of Stems','Number of Bifurcatons',\
            'Number of Branches','Number of Tips','Overall Width','Overall Height',\
            'Overall Depth','Average Diameter','Total Length','Total Surface','Total Volume',\
            'Max Euclidean Distance','Max Path Distance','Max Branch Order','Average Contraction',\
            'Average Fragmentation','Average Parent-daughter Ratio','Average Bifurcation Angle Local',\
            'Average Bifurcation Angle Remote','Hausdorff Dimension']
        csv_writer.writerow(t)
        for rows in feature_map:
            csv_writer.writerow(rows)
        f.close()

def Init_PCA(data,method="None"):
    X = data.astype(np.float64)
    if method=="z-score":
        for i in range(0,np.shape(data)[1]):# z-score
            X[:,i]=(X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
    for i in range(0,np.shape(data)[1]):
        if np.max(X[:,i])-np.min(X[:,i])==0:
            X[:,i]=1
        else:
            X[:,i]=(X[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
    pca = PCA(n_components=3)
    newX = pca.fit_transform(X) 
    cc=pca.components_[0:3,:]
    return newX,cc

def GetLength(A,B):
    temp=0
    for i in range(0,len(A)):
        temp+=(A[i]-B[i])*(A[i]-B[i])
    return math.sqrt(temp)

def GetMST():
    feature_name=[
    'Number of Stems','Number of Bifurcatons','Number of Branches',
    'Number of Tips','Overall Width','Overall Height','Overall Depth',
    'Total Length','Total Volume','Max Euclidean Distance',
    'Max Path Distance','Max Branch Order','Average Contraction',
    'Average Parent-daughter Ratio','Average Bifurcation Angle Local',
    'Average Bifurcation Angle Remote','Hausdorff Dimension']
    with open('Feature.csv')as f:
        feature_data = list(csv.reader(f))
        del feature_data[0]
        f.close()
        data=np.array(feature_data)
        data=data[:,[0,9,10,11,12,13,14,15,17,19,20,21,22,23,25,26,27,28]]
        
    temp_data=data[:,1:].astype(np.float64)  
    newX,component=Init_PCA(temp_data)  
    
    temp_data=newX[:,0:3]
    local_data=newX[:,0:3]

    plt.close()
    ana_data=temp_data.astype(np.float64)
    ax=sns.displot(ana_data,stat="probability",kde=True,alpha=0.4,edgecolor=None,legend=False)
    ax.figure.set_size_inches(8,4)
    mean_t=np.mean(ana_data,0)
    std_t=np.std(ana_data,0)
    plt.legend(labels=["PCA1\nMean:"+str(round(mean_t[0],4))+" Std:"+str(round(std_t[0],4)),
                       "PCA2\nMean:"+str(round(mean_t[1],4))+" Std:"+str(round(std_t[1],4)),
                       "PCA3\nMean:"+str(round(mean_t[2],4))+" Std:"+str(round(std_t[2],4))],fontsize = 14)
    plt.tight_layout()
    plt.savefig('Histogram_PCA.jpg',dpi=300)


    weight_matrix=np.zeros((len(temp_data),len(temp_data)))
    for i in range(0,len(temp_data)):
        for j in range(i,len(temp_data)):
            weight_matrix[i,j]=GetLength(temp_data[i].astype(np.float64),temp_data[j].astype(np.float64))
            
    G = nx.Graph()
    for i in range(0,len(temp_data)):
        for j in range(i+1,len(temp_data)):
            G.add_edge(i, j, weight=weight_matrix[i,j])
    print("number of edges:", G.number_of_edges())   
    
    T=nx.minimum_spanning_tree(G) # 边有权重
    
    mst=nx.minimum_spanning_edges(G,data=False) # a generator of MST edges
    edgelist=list(mst) # make a list of the edges
    
    edgelist_t=sorted(edgelist)
    nodelist={}
    nodelist[edgelist[0][0]]=-1

    while len(edgelist_t)>0:
        for edge in edgelist_t:
            if edge[0] in nodelist.keys():
                nodelist[edge[1]]=edge[0]
                edgelist_t.remove(edge)
                continue
            elif edge[1] in nodelist.keys():
                nodelist[edge[0]]=edge[1]
                edgelist_t.remove(edge)
                continue
    swc_file=[]
    for i in range(0,len(local_data)):
        temp=[i+1,1]
        cc=local_data[i,:]
        temp.extend(cc.tolist())
        temp.append(1)
        temp.append(nodelist[i]+1)
        swc_file.append(temp)
   
    contents=[]
    for i in swc_file:
        temp=''
        for j in i:
            temp+=str(j)
            temp+=' '
        temp=temp[0:-1]
        temp+='\n'
        contents.append(temp)
    f=open('MST.swc','w+')
    f.writelines(contents)
    f.close()

def Init_Data(data):
    X = data.astype(np.float64)
    for i in range(0,np.shape(data)[1]):
        if np.max(X[:,i])-np.min(X[:,i])==0:
            X[:,i]=1
        else:
            X[:,i]=(X[:,i]-np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
    return X

def GetLLE():
   with open('Feature.csv')as f:
       feature_data = list(csv.reader(f))
       del feature_data[0]
       f.close()
       data=np.array(feature_data)
       data=data[:,[0,9,10,11,12,13,14,15,17,19,20,21,22,23,25,26,27,28]]
   
   temp_data=data[:,1:].astype(np.float64)  
   train_data=Init_Data(temp_data)
   LLE = manifold.LocallyLinearEmbedding(n_neighbors = 5, n_components = 3,  method='standard')

   local_data=LLE.fit_transform(train_data)
   temp_data=local_data

   plt.close()
   ana_data=temp_data.astype(np.float64)
   ax=sns.displot(ana_data,stat="probability",kde=True,alpha=0.4,edgecolor=None,legend=False)
   ax.figure.set_size_inches(8,4)
   mean_t=np.mean(ana_data,0)
   std_t=np.std(ana_data,0)
   plt.legend(labels=["LLE1\nMean:"+str(round(mean_t[0],4))+" Std:"+str(round(std_t[0],4)),
                      "LLE2\nMean:"+str(round(mean_t[1],4))+" Std:"+str(round(std_t[1],4)),
                      "LLE3\nMean:"+str(round(mean_t[2],4))+" Std:"+str(round(std_t[2],4))],fontsize = 14)
   plt.tight_layout()
   plt.savefig('Histogram_LLE.jpg',dpi=300)

   contents=['##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b'+'\n']
   count=1
  
   for i in range(0,len(local_data)):
       x=local_data[i]
       t=str(count)+',,,,'+str(x[2]*10)+','+str(x[0]*10)+','+str(x[1]*10)+',0.000,0.000,0.000,0.0001,0.000,,,,255,0,0\n'
       contents.append(t)
       count+=1
   f=open('LLE.apo',"w+",newline='')
   f.writelines(contents)
   f.close()
    
if __name__ =='__main__':
    Init()
    run__pool()
    GetFeatureCSV()
    GetMST()
    GetLLE()
    
 

    
    
    
    
    
    
    