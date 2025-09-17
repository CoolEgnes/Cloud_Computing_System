# Training and Inferencing the Deep Learning Models on FaaS
This is the code for the Cloud Computing System Coursework 2.  
In this repository, two machine learning models were implemented on two serverless computing platforms.   
The commercial platform is Microsoft Azure, all codes executed on this platform are stored in the folder "Commercial platform-Azure". 
There are three subfolders under this folder.   
The first one is linear-regression-vscode, this folder contains the code for training a linear regression model for fitting a formula, and was developed using VSCode.   
The second folder is lr-inference-vscode, which contains the code of using the trained linear regression model for inferencing. This code was also developed using VSCode.  
The third one is resnet-vscode, this folder stores the code for training a resnet50 model for image classification, which is a subset of the Cifar10 dataset, with 1000 randomly sampled images. The code was also developed using VSCode.  
The open-source platform is OpenFaaS, whose codes are stored in the folder, "Opensource platform-OpenFaaS".  
Sub-folders in this folder follow a similar pattern in the "Commercial platform-Azure".
The project successfully verified that:
1.For simple ML models like linear regression, training on FaaS is much more efficient than on local machine. This is consistent with the arguments from the literature review.  
2.When inferencing on the serverless computing platforms, with the increasement of invocations, in this work represented by the number of threads, the throughput and the average response time grew linearly on both platforms. The CPU utilisation followed a roughly similar pattern, while on OpenFaaS a wider range was observed. The observation is consistent with the statements in the literature review.  
3.By comparing the performance of two platforms, it is obvious that Azure functions are more efficient than OpenFaaS functions. This is consistent with the statements in the literature review.
