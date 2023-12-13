# HAI-2023-Object-Detection

## 초기 세팅
코드 구현 환경 : wsl ubuntu-22.04 && Windows 10, miniconda 가상환경 사용

필요한 패키지 설치 방법 : ```pip install -r requirements.txt```

혹여 누락된 패키지가 있다면, 해당 패키지는 ```pip install (package)```를 통해 설치하시면 됩니다.

## 데이터셋 다운로드
아래의 데이터셋을 사용했습니다.

https://universe.roboflow.com/fridgeingredients/fridge-object/dataset/3

위 링크에서 데이터셋(roboflow.zip)을 다운로드 받은 후, 
main.py가 있는 경로에 dataset 폴더를 생성한 뒤 해당 폴더 안에 roboflow.zip 파일을 압축해제하시면 됩니다.


## 실행방법
trian : ```bash ./scripts/train.sh```

eval : ```bash ./scripts/eval.sh```

draw(bbox 그려진 이미지 그리기) : ```bash ./scripts/draw.sh```

use_model(모델을 사용해 image를 image with bbox와 식재료 리스트로 저장) : ```bash ./scripts/use_model.sh```


인자를 직접 설정하고자 할 때는, python main.py --help를 통해 인자에 대한 설명을 볼 수 있습니다.


## 모델 사용 관련
input_image 폴더 내의 마지막 이미지를 model에 넣어 bbox를 그린 이미지와 식재료 리스트(집합)을 outputs 폴더에 저장합니다.

bbox를 그린 이미지는 "bbox_(model에 넣은 이미지 이름)"으로 저장되고, 식재료 리스트는 "ingredients_(model에 넣은 이미지 이름)"으로 저장됩니다.

현재 구현한 코드는 한번에 하나의 쿼리만 처리합니다.

즉, 한 명의 유저가 하나의 사진을 input_image 폴더로 보낸 후 Object Detection하는 상황을 가정하여 코드를 작성했습니다.

##  Prediction 결과
![DSC_5774_JPG_jpg rf 1057b37f4a69f82473279c5396311afa](https://github.com/twkang43/HAI-2023-Object-Detection/assets/90116816/199acff9-05d6-4582-8c16-1e59471b583b) | ![bbox_DSC_5774_JPG_jpg rf 1057b37f4a69f82473279c5396311afa](https://github.com/twkang43/HAI-2023-Object-Detection/assets/90116816/be2fabcf-df80-490a-a9a3-7a3438d05da9)
:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:
<원본 이미지>                                                                                                                                                        |<예측 이미지>

- 인식한 식재료 리스트 : 
  `[
  "tomato",
  "corn",
  "flour",
  "carrot",
  "milk",
  "strawberry",
  "lime"
]`
