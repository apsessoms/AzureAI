# Using Question Answering with Language Studio

## Overview
This guide walks you through creating and training a knowledge base using Azure AI Language Studio’s Question Answering feature. You'll build a knowledge base using Margie’s Travel FAQ document and test its functionality.

## Create a Language Resource
To begin, you need a Language resource in Azure.

1. Open the [Azure portal](https://portal.azure.com) and sign in.
2. Click **+Create a resource**, search for **Language service**, and select **Create**.
3. In the "Select Additional Features" page:
   - **Default features**: Keep as is.
   - **Custom features**: Select **Custom question answering**.

![alt text](https://i.imgur.com/2epuRk8.png)

   - Click **Continue**.
4. Fill in the "Create Language" form:
   - **Subscription**: Select your Azure subscription.
   - **Resource group**: Choose or create one.
   - **Region**: Select a region (e.g., East US 2).
   - **Name**: Provide a unique name.
   - **Pricing tier**: Select **S (1K Calls per minute)**.
   - **Azure search region**: Choose any available location.
   - **Azure search pricing tier**: Choose **Free F (3 Indexes)** (or **Basic** if unavailable).
   - **Responsible AI Notice**: Check the box.
5. Click **Review + Create**, then **Create** and wait for deployment.

### Note
If a free-tier Azure Cognitive Search resource is already in use, you may need to select another pricing tier.

## Create a New Project in Language Studio
1. Open [Language Studio](https://language.azure.com) and sign in.
2. If prompted, select:
   - **Azure directory**
   - **Azure subscription**
   - **Language resource**
3. If not prompted:
   - Click **Settings (⚙)**.
   - Go to **Resources**.
   - Select your Language resource and **Switch resource**.
   - Return to the Language Studio home page.
4. Click **Create new** → **Custom question answering**.

![alt text](https://i.imgur.com/GdEF889.png)

5. On the "Choose language setting" page, select **I want to select the language when I create a project**, then click **Next**.
6. On the "Enter basic information" page:
   - **Language resource**: Select your resource.
   - **Azure search resource**: Select your search resource.
   - **Name**: Enter `PendersTravel`.
   - **Description**: "A simple knowledge base".
   - **Source language**: English.
   - **Default answer**: "No answer found".
   - Click **Next** → **Create project**.

![alt text](https://i.imgur.com/qOAIBqz.png)

7. On the **Manage sources** page:
   - Click **+ Add source** → **URLs**.
   - In the **Add URLs** box:
     - **URL name**: `MargiesKB`
     - **URL**: `https://raw.githubusercontent.com/MicrosoftLearning/mslearn-ai-fundamentals/main/data/natural-language/margies_faq.docx`
     - **Classify file structure**: Auto-detect
   - Click **Add all**.

## Edit the Knowledge Base
You can edit or add custom Q&A pairs to the knowledge base.

1. Expand the left panel and select **Edit knowledge base**.
2. Click **+** to add a new Q&A pair.
   - **Question**: `Hello`
   - **Answer**: `Hi`
   - Click **Done**.
3. Expand **Alternate questions** → **+ Add alternate question** → Enter `Are you there? Whats up? & Hello?`.
4. Click **Save**.

## Train and Test the Knowledge Base
1. Click **Test** and a window pane will appear to the right. 

![alt text](https://i.imgur.com/dVYMPHi.png)

![alt text](https://i.imgur.com/7Fcl5s4.png)

You can see a test chat window where you can enter the three different questions you added to see the response. 

2. Enter `Hi` in the test pane. The response should be `Hi`.
3. Enter `I want to book a flight.`. The response should match the FAQ.
4. Try `How can I cancel a reservation?`.
5. When done, close the test pane.

## Deploy the Knowledge Base
1. Click **Deploy knowledge base** in the left panel.
2. Click **Deploy**.
3. Confirm deployment.

## Cleanup (Optional)
If no longer needed, delete resources to avoid charges:

1. Open the [Azure portal](https://portal.azure.com).
2. Select the resource group containing the created resource.
3. Delete the resource.

## Additional Information
To learn more about Azure Question Answering, refer to the official [documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/language-service/question-answering/).
