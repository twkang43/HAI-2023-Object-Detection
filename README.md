# HAI-2023-Object-Detection

## 초기 세팅
```pip install -r requirements.txt```

현재 requirements.txt가 완전하지 않습니다. (pytorch_lightning, timm 등 누락)

누락된 패키지는 ```pip install (package)```를 통해 설치하시면 됩니다.


## 실행방법
trian : ```bash ./scripts/train.sh```

eval : ```bash ./scripts/eval.sh```

draw(bbox 그려진 이미지 그리기) : ```bash ./scripts/draw.sh```


인자를 직접 설정하고자 할 때는, python main.py --help를 통해 인자에 대한 설명을 볼 수 있습니다.
