# Azure AI Text Analytics Guide

## Creating a Project

### Step 1: Generate a Project Name
When setting up a new project, Azure automatically generates a project name. You can keep the suggested name or modify it as needed.

### Step 2: Select an Azure AI Hub
Depending on your previous usage, you may see a list of existing hubs or an option to create a new one. If you see a dropdown with existing hubs, select **Create new hub**, provide a unique name, and proceed by clicking **Next**.

> **Note**: Ensure you have an Azure AI Services resource provisioned in a supported region to complete this setup.

### Step 3: Choose a Location
In the "Create a project" pane, select **Customize**, then choose from the following supported locations:
- East US
- France Central
- Korea Central
- West Europe
- West US

Click **Create** to proceed.

### Resources Deployed
Once the setup is complete, the following resources will be created:
- **Azure AI Services**
- **Azure AI Hub**
- **Azure AI Project**
- **Storage Account**
- **Key Vault**
- **Resource Group**

After deployment, you will be directed to your project's **Overview** page. From the left-hand menu, navigate to **Playgrounds**.

![alt text](https://i.imgur.com/CxMkWIc.png)


## Extracting Named Entities
1. On the **Playgrounds** page, select **Language playground**.
2. Click **Extract information**, then choose **Extract named entities**.

![alt text](https://i.imgur.com/OGNGuK5.png)

3. Copy and paste the following sample text:

```plaintext
Nice place, great service
Charlotte Motor Speedway, Concord, NC
5/6/2018
The 536 section was an excellent place to watch the race from and you could walk around and experience the V8 majesty from several different places just because you're at the track. The Roval 400 is a treat because you get to see the cars dive into and power out of some excellent corners!
Neat place, even if not on a race day. Things of interest to look at on the track. Food and service at the Speedway Cafe were great.

Parking was abundant around the track and overall it was a treat to be here.
```

4. Click **Run** to process the text.
5. Review the extracted entities in the **Details** section, which includes entity types and confidence scores.

![alt text](https://i.imgur.com/zfosWNc.png)

## Extracting Key Phrases
1. In the **Language playground**, go to **Extract information**, then choose **Extract named entities**.
2. Copy and paste the following text:

```plaintext
Fast lines with poor management
Bank of America Stadium, Charlotte, NC
1/22/2022
This stadium is very outdated and lacks amenities common at other stadiums. It is located in the heart of Uptown near Truist Bank and Romere Bearden Park.  The seat numbers are barely visible from being worn down. The men’s bathrooms are poorly lit, have two stalls total, toilets at ankle height, sinks that continuously run, and no heating. Beverage vendors loiter outside bathrooms and block section entrances. Concessions was poorly designed and managed with long lines and menu options being unavailable for 30+ minutes requiring a refund. Security on entry seemed inconsistent with some lines moving very fast and others taking much longer.
```

3. Click **Run** and review the extracted key phrases and in the **Details** section. The confidence score indicates the likelihood of the information actually belongs to that category.

+ It identifies the most relavent/ important words or phrases that summarize the content. 
    + **"amenities", "America Staidum," "Charlotte," "security," "refund"** were words that carried significant meaning in the text.
    + **"poor management," "long lines" and "bathrooms"** picked up on major issues in the text.

![alt text](https://i.imgur.com/uLzvwCw.png)

The different entities seen in the screen shot are:
+ Address
+ DateTime
+ Event
+ Structural 



## Summarizing Text
1. In the **Language playground**, go to **Summarize information**, then select **Summarize text**.
2. Copy and paste the following text:

```plaintext
Take time to view exhibit
Mint Museum, Charlotte, NC
12/5/2024
If you haven't gone to see Southern Modern,  you owe it to yourself to go. The exhibit will be at the Mint until Feb 2, 2025. 100 paintings from the Mid-Century period when American artists re-shaped the art world. They reflect the time in our country's history full of struggle as well as the personal stories and testaments to humanity's successes and failures. Take time to view the exhibit. Slow view the works. Spend more than 3 minutes looking at a painting. Visually grasp the subtleties and nuances. The information for each work of art was written to help the viewer understand the significance this collection has in society today. Charlotte and the Carolinas is fortunate to have Southern Modern at the Mint Museum.
```



3. Click **Run** and review the summary. The **Details** section provides ranked sentences based on relevance.

![alt text](https://i.imgur.com/V1Q43zs.png)

+ The **extractive summary** selects and gives confidence scores to the most relevant sentences from the text. The **abstractive summary** generates a new summary based on the text content by rephrasing and combining sentences. 

## Cleaning Up Resources
To avoid unnecessary charges, delete unused resources when you are finished.

1. Visit the [Azure portal](https://portal.azure.com).
2. Navigate to the **Resource Group** that contains your project.
3. Select the resources you want to remove.
4. Click **Delete**, then confirm by selecting **Yes**.

## Learn More
For further details on Azure AI Text Analytics, visit the [Language service page](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/).
