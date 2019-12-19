This repository holds the source code and raw experimental results of DASFAA 2020 paper 34. This repository is for reviewing purposes only. All identity information has been hidden.

# How to use the code

The main file in this repository is HBOP.py under the Classification folder. It implements the HBOP algorithm and all its variants (see the "Impact of Design Choices" section in our paper). Note that only the full version of HBOP is timed. Also, our raw experimental results are in the Results folder.

To use the code, please keep all the packages and files in this repository in a single folder, and make sure that you have Python 3 and all the packages we have used installed. Next, please take the following two steps. 

1. Change the path in line 18 of HBOP.py to the mother folder's path. 

2. Run the following command in your command council.

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

To use the code, please keep all the packages and files in this repository in a single folder, and make sure that you have Python 3 and all the packages we have used installed. Next, please take the following two steps.

1. Change the path in line 13 of BOPF.py to the mother folder's path. 

2. Run the following command in your command council.

    python.exe [full path of BOPF.py] [dataset name] [runId] [mother path of the dataset] [full path of the folder to save the results]

Here runId is used to distinguish between different runs on the same dataset.

For example, suppose you have saved the entire repository at G:\HBOP, and you wish to run BOPF on the sample dataset (FaceFour) included in our repository and save the outputs in the Results folder. In this case, please run the following command.

    python.exe G:\HBOP\Classification\BOPF.py FaceFour 0 G:/HBOP/Data G:/HBOP/Results  

Running BOPF.py will result in the following output files.

1. accuracy_[dataset name]_BOPF.[runId].txt

    This file shows the accuracy of BOPF
    
2. time_[dataset name]_BOPF.[runId].txt

    This file shows the running time of BOPF. The three outputs are the dataset name, training time and classification time per example.

# Why have we omitted some datasets?

We have used datasets from the latest version of the UCR Archive 

    Hoang Anh Dau, Eamonn Keogh, Kaveh Kamgar, Chin-Chia Michael Yeh, Yan Zhu, Shaghayegh Gharghabi , Chotirat Ann Ratanamahatana, Yanping Chen, Bing Hu, Nurjahan Begum, Anthony Bagnall , Abdullah Mueen and Gustavo Batista (2018). The UCR Time Series Classification Archive. URL https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    
The UCR Archive now consists of 128 datasets, among which we have used 106. We have omitted the remaining 22 datasets for the following reasons.

First, we have omitted all 11 variable-length datasets, namely those with different time series lengths. This is because for the three 1NN baselines, we have obtained their classification error rates from the UCR archive webpage. For variable-length datasets, these values are obtained by running 1NN algorithms after pre-padding shorter time series with low-amplitude random values. For fairness of comparison, we would have to take the same measure for non-1NN methods if we were to include these datasets. However, this brings about a problem. Padding values have different effects on 1NN and non-1NN based methods (at least for some implementations of these methods). For example, when z-normalizing time series, 1NN-based methods z-normalize the entire time series, meaning that these padded low-amplitude values will have limitted effect on ensuing classification. However, HBOP and BOPF z-normalize each subsequence individually, meaning that when they (at least the BOPF implemented by its original authors)  extract subsequences made up of these padded values, their low amplitudes tend to be over-amplified by z-normalization. This means that if we were to pad the shorter time series, we would have to take into account the different effects of padding when comparing different methods, which distracts the main goal of our expriments, namely to only evaluated the classification methods themselves. To this end, we have chosen to omit the variable-length datasets.

It is worth noting that while we have omitted the variable-length datasets for consistency in exprimental comparisons, our HBOP itself can be directly applied to such datasets.

Second, on certain datasets with large sizes (e.g. ElectricDevices) and/or high numbers of classes (e.g. FiftyWords), our current implementation can encounter memory failures. We believe that this is more likely due to implementation issues, NOT the innate drawback of our method. For example, for X-means, we have used the Python package PyClustering:
   
   Novikov, A., 2019. PyClustering: Data Mining Library. Journal of Open Source Software, 4(36), p.1230. Available at: http://dx.doi.org/10.21105/joss.01230
   
To run X-means with this package, we keep feature vectors of all time series in a single Numpy array, which can be demanding to memory. We are currently exploring possible alternatives to this approach, e.g. using a hybrid of Python and C++ when using this package.

Third, on certain datasets, it can take too long for our Python implementation to run the full HBOP algorithm. While our HBOP itself is efficient (as is evidenced by its RELATIVE running time to BOPF), unfortunately our Python implementation is not (as is evidenced by our long ABSOLUTE running time). To this end, we have chosen to run the small- and medium-sized datasets first, and then the large ones to make sure that we can run on as many datasets in as short time as possible. This means that several large datasets have been left out, which we are currently running our code on. We will update our results once we have obtained our performances on these datasets.

In conclusion, while we have omitted 22 datasets for various reasons, we believe that our experiments are currently substantial enough, considering the fact that we have used over 100 datasets. However, we are currently doing our best to obtain results on the remaining datasets to make our results more compelling.

# How did we obtain the baseline experimental results?

In our paper, we compared the classification accuracy of our HBOP method with three 1NN algorithms, FS, BOSS-VS and BOPF. The classification error rates of the 1NN algorithms were drawn from the UCR archive webpage. For FS, BOSS-VS and BOPF, we used the code provided by the original authors to obtain their results. In addition, for BOPF we also used our own Python implementation. The final accuracy for BOPF is the higher one of the results obtained using our code and the original code. All results were rounded to four digits before comparison. 

As with efficiency, we implemented BOPF with Python and used this implementation to obtain the training and classification time of BOPF. Note that the running time obtained this way is much longer than those reported in the original BOPF paper. This is due to implementation bias. For example, the BOPF paper used C++, while we used Python which is known to be slower.

# On BOPF classification time on the Fungi dataset
When comparing online classification time, we have discarded the Fungi dataset, for the classification time per example for BOPF on it is abnormally small. On this dataset, while HBOP took 2.263 seconds to classify an example, BOPF only took 0.000024 seconds. The unusually short time is liekly due to the fact that in the training phase, the number of words selected by BOPF is very low (probably zero). This is evidenced by BOPF's poor accuracy (when implemented in Python) on this dataset (only 0.0376). However, this is likely due to implementation bias, as using the C++ code by the original authors can bring about a high accuracy of 0.979. For fairness to BOPF, we have adopted the latter value in our experiments concerning accuracy comparisons.
