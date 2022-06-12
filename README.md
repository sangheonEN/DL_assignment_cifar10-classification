# DL_assignment_cifar10-classification

**1.	Introduction**
일반적으로 이미지 분류는 심화학습 기초 수업에서 습득한 딥러닝 모델 구축 방법론을 기반으로 CIFAR-10 데이터를 활용한 세 가지 Convolution Neural Network(CNN) 이미지 분류 모델을 구현하고 Data Augmentation, 여러 Learning Rate Scheduling 방법론을 적용하여 비교 검증 한다.

**2.	Proposal Method**
1)	Data Processing
-	Train, Valid, Test set
-	Data Normalization (mean, std)
-	Augmentation (Random Rotation, Random Crop, Horizontal Flip)
2)	CNN Model 
-	ResNet50_32x4d
-	EfficientNet
-	MobileNetV3
3)	Early Stopping 
-	Validation Loss 기준으로 이전의 Epoch의 loss값이 이후의 Epoch loss 값보다 작으면 count하여 20회 누적 시 모델 종료.
4)	Optimizer SGD -> ADAM 변경
5)	Learning Rate Scheduling
-	Origin (learning rate value변화 없음.)
-	Step LR (step: 20, gamma: 0.2)
-	Warm-up LR (warm step: 5. after warm-up, return initial learning rate)
-	Lambda LR (0.95epoch  0.95에 epoch 승수 만큼 변화) 
