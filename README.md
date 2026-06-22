# Bee YOLO Detector

Flask와 Ultralytics YOLO를 사용하는 6종 벌 객체 탐지 웹앱입니다.

## 포함된 기능

- `.pt` YOLO 모델 업로드 및 로드
- 단일 이미지 예측
- `test/test.json` 기반 mAP 평가
- 탐지 박스가 그려진 결과 이미지 표시

## 프로젝트 구조

```text
.
├── app.py
├── run.py
├── make_pred_labels.py
├── requirements.txt
├── templates/
├── static/
├── models/          # 웹에서 업로드한 .pt 모델 저장 위치
├── model_storage/   # 보조 스크립트용 모델 저장 위치
└── test/            # mAP 평가용 test.json 및 이미지 위치
```

대용량 모델 파일, 테스트 이미지, 평가 결과, 가상환경은 Git에 포함하지 않습니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux에서는 가상환경 활성화 명령만 다릅니다.

```bash
source .venv/bin/activate
```

CUDA용 PyTorch가 필요하면 사용 중인 CUDA 버전에 맞춰 PyTorch를 별도로 설치하세요.

## 실행

```bash
python run.py
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:5002
```

## 모델 파일

모델 파일은 GitHub에 포함하지 않습니다. 웹 화면에서 `.pt` 파일을 업로드하면 `models/` 폴더에 저장되고 즉시 로드됩니다.

보조 스크립트 `make_pred_labels.py`를 사용할 경우 기본 모델 위치는 다음과 같습니다.

```text
model_storage/best_yolo26m_8276_img768_map873.pt
```

## mAP 평가 데이터

mAP 평가를 사용하려면 `test/` 폴더에 다음 파일들을 직접 넣어야 합니다.

```text
test/
├── test.json
└── *.jpg, *.jpeg, *.png
```

평가를 실행하면 `converted_test/`와 `map_results/`가 자동 생성됩니다. 이 폴더들은 Git에 포함하지 않습니다.

## 참고

프로젝트 내부 경로는 현재 프로젝트 폴더 기준으로 계산됩니다. 따라서 프로젝트 폴더를 `C:`, `D:`, 다른 사용자 폴더 어디에 옮겨도 같은 구조만 유지하면 실행할 수 있습니다.
