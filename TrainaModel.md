# Step-by-Step Guide to Create an Azure Machine Learning Model

## Prerequisites
- An active Azure subscription
- Azure Machine Learning workspace

## Introduction
In this guide, you'll use Azure Machine Learning Studio's automated machine learning feature to train and evaluate a machine learning model. This tool simplifies the process by automatically selecting the best algorithms and settings for your data. You'll explore its results and decide if the model is ready for deployment. We will use a dataset from Microsoft. 
## Steps

### 1. Sign in to the Azure Portal
- Go to [Azure Portal](https://portal.azure.com/)
- Sign in with your Azure account credentials

### 2. Create a Machine Learning Workspace
- In the Azure portal, select **Create a resource**.
- Search for **Machine Learning** and select it.
- Click **Create**.
- Fill in the required details:
    - **Workspace name**: Enter a unique name for your workspace.
    - **Subscription**: Select your Azure subscription.
    - **Resource group**: Create a new resource group or select an existing one.
    - **Location**: Choose the region closest to your users.
- Click **Review + create** and then **Create**.

### 3. Launch Azure Machine Learning Studio
- Once the workspace is created, go to the resource.
- Click on **Launch studio** to open Azure Machine Learning Studio.

### 4. Create a Automated ML job
- In the Azure Machine Learning Studio, select **Automated ML** from the left-hand menu
- Under **Basic** settings  settings:
    + **Job Name**: Keep the prepopulated name as it is. 
    + **Experiment Name**: 

![alt text](https://i.imgur.com/OeTkypY.png)

## 5. Task type & data
- Select task type: **Regression**
- Select dataset: 
    + Name: bike-rentals
    + Description: Historic bike rental data
    + Type: Table(mltable)
+ Select Next
+ **Data Source:** 
    + From Local files
+ **Destination Storage Type:**
    + Azure Blob Storage
    + Name: workspaceblobstore
+ Select Next

![alt text](https://i.imgur.com/4rEFcAT.png)

Download this dataset from Microsoft at [https://aka.ms/bike-rentals] & upload it into the Automated ML job. 

![alt text](https://i.imgur.com/oHXFSOF.png)

Click on create and after the dataset is created, click on **bike-rentals** to continue.

![alt text](https://i.imgur.com/hYhe1yX.png)

A preview of the dataset will be shown. Click on **Next** to continue.

**Task Settings**
    + Target Column: **Rentals(integer)

![alt text](https://i.imgur.com/iraGwwq.png)

Click **Save**. Expand the limits section. 

+ Enter the following settings:

![alt text](https://i.imgur.com/BKxn5Mu.png)

Select **Next** to continue. Under **Computer** leave everything as default and click **Next**.

+ Review your Automated ML job information and click **Submit training job**. 

It will take several minutes for the job to complete. Once it is complete, you can view the results of the job.

![alt text](https://i.imgur.com/GuqWdRJ.png)

## 6. Review the model
- Click on the text underneath the **algorithm name** to view its details.

![alt text](https://i.imgur.com/AzDHlWN.png)

+ **Algorithm name**: VotingEnsemble - combines predictions from the models we selected in previous steps (RandomForest & LightGBM)
+ **Ensemble Details**: Click to show more about what was weighted in voting process.
+ **Normalized root mean squared error**: This measures how far off the models predictions are from the actual values. The closer to 0 the better. In this case, 0.08857 is a low error, indicating good performance.
+ **Sampling**: 100% of the data was used to train the model. 
+ **Registered Models**: This trained model has not been saved to a repository for reuse yet. 
+ **Deploy Status**: The model is not deployed into production yet where it can make predictions. 

## 7. Deploy the model
On the **Model** tab, select **Deploy with real time-endpoint** option selected. 

![alt text](https://i.imgur.com/D4tqHkg.png)

Use the following settings:

![alt text](https://i.imgur.com/bvIsd8w.png)


## Test the deployed service

In the left hand menu, select **Endpoints** and then select the **Test** tab.
+ Replace the JSON with the following:

```
   {
  "input_data": {
    "columns": [
      "day",
      "mnth",
      "year",
      "season",
      "holiday",
      "weekday",
      "workingday",
      "weathersit",
      "temp",
      "atemp",
      "hum",
      "windspeed"
    ],
    "index": [0],
    "data": [[1,1,2022,2,0,1,1,2,0.3,0.3,0.3,0.3]]
  }
 }
```
+ Click the **test** button & review the results that include the predicted value for the number of bike rentals. In my case, the predicted value was 334.6750617435755. 


### Conclusion
You have successfully created, trained, evaluated, and deployed a machine learning model using Azure Machine Learning.

## Clean up
- Delete the Azure Machine Learning workspace to avoid incurring unnecessary costs.
- Delete the resource group associated with the workspace in the Azure portal.