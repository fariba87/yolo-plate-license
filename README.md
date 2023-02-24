# yolo-plate-license
In this project I am going to detect and recognize the Iranian cars license plate.
[ANPR(automatic number plate recognition) application]:

the overall procedure consists of:

1) plate detection
2) plate tracking[if input is video]
3) plate number recognition(OCR)
    
As every ML project, we should consider data and model
1) Data 
    
    (as supervised, we need both data and label):

      1.1) 	for detection:

        input: car image 
	    label: bounding boxes
	    Note 1: There are many labeling tools (CVAT, LabelImg, VoTT), and new emerging online tool: roboflow[either for annotating a raw dataset or converting the existing annotations to appropriate one for the model]
	    Note 2: Yolo annotation for bounding boxes: (xcenter, ycenter, width, height)
	    Note 3: yolo predict the offset for bounding boxes
	    Note 4: yolov8 is anchor free
	
      1.2) for recognition:
	    
        input: a rectangle image consisting of sequence of numbers and alphabel        
		Note: we can assume the orientation of text line is horizontal,otherwise an alignment is needed as preprocessing
		label: text(sequence of characters)
2) Model:

	2-1) Detection:
   
       traditional : HOG, contour detection
       deep learning based:
               two stages: RCNN, FastRCNN
               one stage : YOLO series, SSD, retinanet
                   1) YOLO v7
                   2) also we will try the new released YOLOv8( January 2023) as it outperformes the previous models (also still is in developing stage)
   2-2) Tracking:

       traditional: kalman filter , particle filter, 
       deep learning based:
            SORT
            deepSORT
   2-3) Recognition:

       traditional: template based
       deep learning based:
               . using already existing packages for OCR(supporting persian/arabic languages) like easyOCR, tesseract, PP-OCR , 
               . writing a model from scratch for this perpose:
                     most developed models in this area are either based on CTC, attention, transformer or the composition of these architectures.
		
			
##  Implementation: 


  1) using google colab (free GPU)	 
  2) using local machine's GPU:

	1) create a new venv[in windows]: py -m venv yolov7env
	2) activate : yolov7env\Scripts\activate
	3) clone yolov7 repo : git clone https://github.com/WongKinYiu/yolov7.git
	4) change dir : cd yolov7
	Note: since I want to train model on GPU i should change the torch installation(to gpu)-> modify the requirements.txt
	5) open requirements.txt and delete these two files: 
		#torch>=1.7.0,!=1.12.0
		#torchvision>=0.8.1,!=0.13.0
	6) create another requirement_gpu.txt file and add the following lines: 
		-i https://download.pytorch.org/whl/cu113
		torch==1.11.0+cu113
		torchvision==0.12.0+cu113
	7) install all requirements by: 
		pip install -r requirements.txt
		pip install -r requirements_gpu.txt
		pip install requiremet_extra.txt [for tracking and ocr]
	8) write python on cmd and check the availability of cuda:
		python
		import torch
		torch.cuda.is_available()
		if yes congratulation :) --> quit()  python
	9) run the command line for training model 
	



https://learnopencv.com/understanding-multiple-object-tracking-using-deepsort/#:~:text=DeepSORT%20can%20be%20defined%20as,offline%20just%20before%20implementing%20tracking.
https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/
https://www.section.io/engineering-education/license-plate-detection-and-recognition-using-opencv-and-pytesseract/


### Extra Notes
1) what if i want to code in tensorflow?
        
        my answer:as the github repo for yolo is in pytorch, it is better to train the model with command line in torch, save the model(.pt) and export it to .onnx and then convert it to tensorflow model
	    then i can use onnx, onnxruntime to do inference on new images.
2) what if aspect ratio of rectangle is too high or large scale images[it is not the case for license plates] and small scale objects?
	    
        my answer:if using YOLOv7, try to integrate sahi algorithm[vision-library-for-performing-sliced-inference-on-large-images-small-objects:https://github.com/obss/sahi] to yolov7
		yolov8 already solve this problem and no need to sahi
3) challenges in tracking(fast moving car):
 
	    i think deepSORT is for realtime and already solve it
4) is there any improvement in yolo?
	
        my answer:i searched and found by integrating transformer architecture on yolo backbone, there would be improvement
5) challenge of limited dataset size:
	
        my answer: yolo models already use some augmentations like mosaeic,..
	 	we did some augmentation while using roboflow, and we can use more of those available options
		can use GAN to generate images? i dont know if it is applicable for this problem

### How to deploy it?
1) on mobile device: convert model by tflite and use android studio for android coding and swift for iOS coding
2) on server : convert to tf.serving
3) onwindows: you can use c# and Microsoft.ML
4) on openVINO
5) on jetnano	
6) an API with django (python web framework API)
7) kerize your application
