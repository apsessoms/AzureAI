# Azure Face Service
## Introduction
In this guide, you'll learn how to use Azure Face Service to detect faces in images. Azure Face Service is a powerful tool that can be used to detect faces in images and extract valuable insights from them. 

## Use Case 
In this fictional scenario, Pender's Unniversity wants to modernize its attendance tracking system. Instead of using traditional roll calls or ID card swipes, they decide to implement a facial recognition system that detects student faces when they enter the classroom.

To do this, they can use Azure Face Service to test face detection capabilities. The system will capture images from a classroom camera, detect faces, and return bounding box coordinates for each student. 

## Create Azure AI Resource

Go to the [Azure Portal](https://portal.azure.com/) and click on ** Create a resource**. Search for Azure AI Services Plan. Use the following settings:

- **Resource Group**: Create a new resource group or select an existing one.
- **Region**: Choose the region closest to your users. Mine is East or East US 2. 
- **Name**: Enter a unique name for your resource.
- **Pricing Tier**: Choose the pricing tier that best fits your needs.

Click **Review + Create** and then **Create**.

While that is deploying, open the **Vision Studio** [here](https://portal.vision.cognitive.azure.com/?azure-portal=true/). Click on **view all resources** under the **Getting started** section. 

![alt text](https://i.imgur.com/ZER1oz0.png)

Select the resource we just created and click **Select as default resource**.

## Detect Faces 

Go back to **Vision Studio** and select the **Face** tab followed by **Detect Faces in an image**.

Use your own image or use Microsofts [images](https://aka.ms/mslearn-detect-faces) and observe the detected faces in the image. Also observe how the data is returned in the **detected attributes** section.

![alt text](https://i.imgur.com/wU5JetC.png)

**faceRectangle** is the bounding box around the face. 

    + width: The width of the face is 211 pixels.
    + height: The height of the face is 284 pixels.
    + left: The leftmost pixel of the face is 185 pixels from the left of the image
    + top: The topmost pixel of the face is 63 pixels from the top of the image.

**faceLandmarks** have x,y coordinates that represent the exact location in pixels within the image. 

    +"pupilLeft": {"x": 231.5, "y": 191.4} ➡️ Left pupil is at (231.5, 191.4)
    + pupilRight": {"x": 329.9, "y": 176.5} ➡️ Right pupil is at (329.9, 176.5)
🧐 Insight: The right pupil is slightly higher (y=176.5) than the left pupil (y=191.4), which might indicate a slight head tilt

**noseTip**

    + "noseTip": {"x": 279.9.5, "y": 232.2}``` ➡️ Nose tip is at (279.9, 232.2)

**mouthLeft**

    + "mouthLeft": {"x": 240.7, "y": 270.7} ➡️ Left corner of the mouth
    + "mouthRight": {"x": 347, "y": 254.2} ➡️ Right corner of the mouth
🧐 Insight: The right side of the mouth is slightly higher (y=254.2) than the left (y=270.7), which might indicate a subtle smirk or tilt

![alt text](https://i.imgur.com/ciLjKbw.png)

+ Each face is highlighted with a **blue bounding box**.
+ All faces except face #2 are detected without masks. Face #2 is marked as "Face mask: other type of mask or occlusion." This means the system detected some form of obstruction, such as glasses, a shadow, or in this case a beard. This is a an example of how to verify presence in the classroom or workplace. 

![alt text](https://i.imgur.com/kwwwLgH.png)

+ Three out of the four individuals are wearing masks.
Even though the person in the back is the furthest away, the system still detected their face. This is an example of how the system can detect if a person if wearing a mask or not.

# Conclusion

In this lab, we learned how Azure AI services can detect faces in images. 

## Delete Resources

+ To delete this resource, go to the [Azure Portal](https://portal.azure.com/). 
+ Click on the resource group that contains the resources you created.
+ Click on the **Delete** button and confirm the deletion.