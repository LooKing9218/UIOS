# Data Availability
Data from JSIEC1000 is available at (https://www.kaggle.com/datasets/linchundan/fundusimage1000). 

Data from RETORCH is available at (https://retouch.grand-challenge.org). 

Data from VOC2012 is available at (http://host.robots.ox.ac.uk/pascal/VOC/voc2012). 

Additional data sets supporting the findings of this study were not publicly available due to the confidential policy of National Health Commission of China,  but are available from the corresponding authors upon reasonable request. 

# The overview of the uncertainty-inspired open set (UIOS) learning for retinal anomaly classiﬁcation.

![Alt Text](Demo/UIOS_gif.gif)

Standard artiﬁcial intelligence (AI) and our proposed UIOS AI models were trained with the same dataset with 9 categories of retinal photos. In testing, standard AI model assigns a probability value ($p_{i}$) to each of the 9 categories, and the one with the highest probability is output as the diagnosis. Even when the model is tested with a retinal image with disease outside the training set, the model still outputs one from the 9 categories, which may lead to misdiagnosis. In contrast, UIOS outputs an uncertainty score ($u$) besides the probability ($p_{i}$) for the 9 categories. When the model is fed with an image with obvious features of retinal disease in the 9 categories, the uncertainty-based classiﬁer will output a prediction result with a low uncertainty score below the threshold $θ$ to indicate that the diagnosis result is reliable. Conversely, when the input data contains ambiguous features or is an anomaly outside of training categories, the model will assign a high uncertainty score above threshold $θ$ to explicitly indicate that the prediction result is unreliable and requires a double-check from their ophthalmologist to avoid misdiagnosis.
