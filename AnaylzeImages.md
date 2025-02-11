# Analyze Images using Azure AI Vision 👓

In this lab, you will learn how to use Azure AI Vision to analyze images. Azure AI Vision is capable of detecting objects, faces, and text in images. In this lab, we will use AI Vision via the [Azure AI Foundry Portal.](https://ai.azure.com/?azure-portal=true&tid=d97e86c8-9d1d-450c-acd1-adbb4f956aa7)

![alt text](https://i.imgur.com/JeAeXhH.png)

## Steps

+ On the Azure AI Foundry Portal, select **Create a project**.
+ Keep the generated project name and hub as it is and click **Create**.

![alt text](https://i.imgur.com/vsZLUTN.png)

Please note that you should **never** have public facing resources in any enterprise environment. I masked that value out as it is not the focus on this lab. I will review that in a later lab. 📝

You should see between 5-6 different resources being provisioned. This can take a few minutes.

![alt text](https://i.imgur.com/EV0LRIh.png)

+ Once the resources are provisioned, click on **AI Services**.

![alt text](https://i.imgur.com/IalA99D.png)

+ Click on **Vision + Document**

![alt text](https://i.imgur.com/NzXGEUl.png)

## Generate Captions for images 🖼️
Select **images** and click on **image captioning**.


![alt text](https://i.imgur.com/fVfxk0v.png)

+ Make sure that you create a workspace in a region that supports the service. At the time of this lab, these regions are currently being supported:

![alt text](https://i.imgur.com/iS3fMcL.png)

Download any picture you want to analyze and upload it to the service. I googled "Serena Williams" and downloaded an image of her. If you want to use Microsofts image, you can use this link: [https://aka.ms/mslearn-images-for-analysis)

+ Drag the image from your local machine to the area where it says "Drag and drop your image here".

![alt text](https://i.imgur.com/AFxweEP.png)

+ You should see the generated caption for the image in the **detected attributes** section. The provides

Go back to the other vision capabilities and select **Dense Captioning**. This will provide various human-readable captions for an image:
    + The content of the image
    + Objects detected in the image
    + Bounding box
    + Pixel coordinates within the image associated with the object

![alt text](https://i.imgur.com/81M179A.png)

+ If you hover over the detected attributes, you will see the bounding box highlighted in the image.

## Tagging Images 🏷️

Extracting tags from images is a common use case for image analysis. Tags are words or phrases that describe the content of an image. Extract tags is based on thousands of recognized objects, living beings, scenes, and actions.

+ Return back to the vision capabilities and select **common tag extraction**.
+ Drag the image from your local machine to the area where it says "Drag and drop your image here".
+ Review the list of tags that are generated for the image as well as the confidence score for each tag in the detected attributes panel. 

![alt text](https://i.imgur.com/JOrEErv.png)


## Detecting Objects in Images 🕵️‍♂️

+ Go back to the other vision capabilities and select **common object detection**.
+ Drag the image from your local machine to the area where it says "Drag and drop your image here".

![alt text](blob:https://imgur.com/de852747-5166-40ae-9086-43351eb4317d)

+ Review the **detected attributes** box and observe the objects and their confidence score. The score is lower because of the pixilation of the image. 

## Conclusion

In this lab, you learned how to use Azure AI Vision to analyze images. You learned how to generate captions for images, tag images, and detect objects in images. Azure AI Vision is a powerful tool that can be used to analyze images and extract valuable insights from them.

## Delete Resources

+ To delete this resource, go to the [Azure Portal](https://portal.azure.com/). 
+ Click on the resource group that contains the resources you created.
+ Click on the **Delete** button and confirm the deletion.