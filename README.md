Cell_Pattern_Analysis_Tool is designed for analysis and visualization of the neuronal morphological feature space, including:
1. Call the "Global Feature Plugin" in Vaa3d to generate data for morphological features
2. Perform Principal Component Analysis(PCA) on the feature data and generate a Minimum Spanning Tree(MST) .swc file that can be viewed in Vaa3d
Perform Locally Linear Embedding(LLE) on the feature data and generate .apo files that can be viewed in vaa3d

## Instructions
1. Put the swc files that need to be analyzed in the "Data" folder, and change the "vaa3d.conf" file to the vaa3d startup path in your computer
2. Execute the analysis.py file using python
(I am using python 3.9.7)  
Executing the file will automatically install the required packages
## Results
* "Feature.csv" shows the result of the features calculated by the "Global Feature Plugin".
* "Histogram_PCA.jpg"/"Histogram_LLE.jpg" are the statistics of the distribution of first three dimensions.
* "MST.swc" is the MST of the PCA results that can be viewed in vaa3d
* "LLE.apo" is the LLE results that can be viewed in vaa3d