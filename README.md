This repository holds the source code and raw experimental results of DASFAA 2020 paper 34. This repository is for reviewing purposes only. All identity information has been hidden.

# How to use the code

The main file in this repository is HBOP.py under the Classification folder. It implements the HBOP algorithm and all its variants (see the "Impact of Design Choices" section in our paper). Note that only the full version of HBOP is timed. Also, our raw experimental results are in the Results folder.

To use the code, please keep all the packages and files in this repository in a single folder, and make sure that you have Python 3 and all the packages we have used installed. Before running the code, first change the path in line 18 of HBOP.py to the mother folder's path. Then, run the following command in your command council.

    python.exe [full path of HBOP.py] [dataset name] [runId] [mother path of the dataset] [full path of the folder to save the results]

Here runId is used to distinguish between different runs on the same dataset.

For example, suppose you have saved the entire repository at G:\HBOP, and you wish to run HBOP on the sample dataset (FaceFour) included in our repository and save the outputs in the Results folder. In this case, please run the following command.

    python.exe G:\HBOP\Classification\HBOP.py FaceFour 0 G:/HBOP/Data G:/HBOP/Results  
    
Running HBOP.py will result in the following output files.

1. accuracies_[dataset name]_HBOP.[runId].txt

    This file shows the accuracies of 12 variants of HBOP, in the order of HBOP-NW, HBOP-NX-NW, HBOP-X-NW, HBOP-SAX-NW, HBOP-SAX-NX-NW, HBOP-SAX-X-NW, HBOP, HBOP-NX, HBOP-X, HBOP-SAX, HBOP-SAX-NX, HBOP-SAX-X.
    
2. time_[dataset name]_HBOP.[runId].txt

    This file shows the running time of the full version of HBOP. The four outputs are the dataset name, training time, classification time per example and accuracy.

Please note that in certain cases, the accuracies of the full version of HBOP in the "accuracies" file and the "time" file can be different. This is likely due to implementation bias. Fortunately, this only happened in very few cases in our experiments. When such an inconsistency occured, we used the accuracy in the "time" file as the final accuracy.

Also, we have included our Python implementation of BOPF in BOPF.py under the Classification folder, which implements the BOPF algorithm proposed in
    
    	Xiaosheng Li, Jessica Lin: Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features. ICDM 2017: 277-286

To use the code, please keep all the packages and files in this repository in a single folder, and make sure that you have Python 3 and all the packages we have used installed. Before running the code, first change the path in line 13 of BOPF.py to the mother folder's path. Then, run the following command in your command council.

    python.exe [full path of BOPF.py] [dataset name] [runId] [mother path of the dataset] [full path of the folder to save the results]

Here runId is used to distinguish between different runs on the same dataset.

For example, suppose you have saved the entire repository at G:\HBOP, and you wish to run BOPF on the sample dataset (FaceFour) included in our repository and save the outputs in the Results folder. In this case, please run the following command.

    python.exe G:\HBOP\Classification\BOPF.py FaceFour 0 G:/HBOP/Data G:/HBOP/Results  

Running BOPF.py will result in the following output files.

1. accuracy_[dataset name]_BOPF.[runId].txt

    This file shows the accuracy of BOPF
    
2. time_[dataset name]_BOPF.[runId].txt

    This file shows the running time of BOPF. The three outputs are the dataset name, training time and classification time per example.

# On the datasets

We have used datasets from the latest version of the UCR archive 

    Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi , Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu, Nurjahan Begum, Anthony Bagnall , Abdullah Mueen and Gustavo Batista (2018). The UCR Time Series Classification Archive. URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    
Note that for datasets with missing values and variable time series lengths, the UCR archive has provided preprocessed versions of them, which is the data we have used in our experiments.

# How did we obtain the baseline experimental results?

In our paper, we compared the classification accuracy of our HBOP method with three 1NN algorithms, FS, BOSS-VS and BOPF. The classification error rates of the 1NN algorithms were drawn from the UCR archive webpage. For FS, BOSS-VS and BOPF, we used the code provided by the original authors to obtain their results. In addition, for BOPF we also used our own Python implementation. The final accuracy for BOPF is the higher one of the results obtained using our code and the original code. All results were rounded to four digits before comparison. 

As with efficiency, we implemented BOPF with Python and used this implementation to obtain the training and classification time of BOPF. Note that the running time obtained this way is much longer than those reported in the original BOPF paper. This is due to implementation bias. For example, the BOPF paper used C++, while we used Python which is known to be slower.

# On BOPF classification time on the Fungi dataset
When comparing online classification time, we have discarded the Fungi dataset, for the classification time per example for BOPF on it is abnormally small. On this dataset, while HBOP took 2.263 seconds to classify an example, BOPF only took 0.000024 seconds. The unusually short time is liekly due to the fact that in the training phase, the number of words selected by BOPF is very low (probably zero). This is evidenced by BOPF's poor accuracy (when implemented in Python) on this dataset (only 0.0376). However, this is likely due to implementation bias, as using the C++ code by the original authors can bring about a high accuracy of 0.979. For fairness to BOPF, we have adopted the latter value in our experiments concerning accuracy comparisons.
