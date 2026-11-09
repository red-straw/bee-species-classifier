# 🐝 Bee Species Classifier

YOLO 기반으로 국내에서 자주 보이는 **벌 6종을 탐지·분류**하고, Flask 웹에서 바로 추론과 mAP 평가를 할 수 있게 만든 프로젝트입니다.

`26-1` · 개인 프로젝트

| 클래스 | 이름 |
|---|---|
| `apis` | 꿀벌 |
| `black` | 등검은말벌 |
| `crabro` | 말벌 (Vespa crabro) |
| `ggoma` | 좀말벌 |
| `jangsu` | 장수말벌 |
| `simil` | 털보말벌 |

## 핵심 내용

- **반자동 라벨링으로 데이터 4.9배 확장**: 직접 라벨링한 1,701장으로 1차 모델을 학습 → 새 이미지 6,616장을 1차 모델로 자동 라벨링 → 빈 라벨·오류 정리(6,575장) → 최종 **8,276장** 데이터셋 구성
- **모델·해상도 비교 실험**: YOLOv8m, YOLO26s, YOLO26m을 이미지 크기(640~960)별로 학습해 비교
- **웹 서비스화**: 모델 업로드/교체, 이미지 추론(박스 시각화), COCO 형식 테스트셋 mAP 평가를 웹에서 수행

## 결과

### 검증셋 (학습 데이터의 20%)

| 모델 | 데이터 | imgsz | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| YOLOv8m | 1,701장 | 832 | 0.899 | 0.853 | 0.923 | 0.768 |
| YOLO26s | 8,276장 | 960 | 0.929 | 0.887 | 0.938 | 0.855 |
| **YOLO26m (최종)** | **8,276장** | **768** | **0.928** | **0.892** | **0.941** | **0.865** |

데이터를 늘린 뒤 mAP50-95가 **0.768 → 0.865**로 크게 올랐습니다.

### 외부 테스트셋 (학습에 쓰지 않은 600장, COCO 형식)

| 모델 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8s (1,701장, 640) | 0.766 | 0.765 | 0.781 | 0.517 |
| YOLOv8m (1,701장, 832) | 0.822 | 0.774 | 0.808 | 0.535 |
| **YOLO26m (8,276장, 768)** | **0.841** | **0.793** | **0.807** | **0.544** |

검증셋과 테스트셋 성능 차이(mAP50-95 0.865 → 0.544)는 테스트 이미지의 촬영 환경이 학습 데이터와 달라서 생긴 것으로 보이며, 도메인이 다른 데이터에 대한 일반화가 다음 개선 과제입니다.

<p>
  <img src="results/figures/results.png" width="49%">
  <img src="results/figures/confusion_matrix_normalized.png" width="49%">
</p>

## 프로젝트 구조

```text
.
├── app/                 # Flask 웹앱 (추론·모델 교체·mAP 평가)
│   ├── app.py
│   ├── run.py
│   ├── templates/
│   └── static/
├── data_tools/          # 데이터 구축 파이프라인
│   ├── auto_label.py    #   1차 모델로 새 이미지 자동 라벨링
│   ├── clean_labels.py  #   빈 라벨/짝 없는 파일 정리 (dry-run 지원)
│   ├── build_dataset.py #   여러 소스 병합 + train/val 8:2 분할
│   └── coco_to_yolo.py  #   COCO(test.json) → YOLO 형식 변환
├── training/
│   ├── train.py         # 학습 (모델·해상도 등 인자로 지정)
│   └── data.yaml
├── evaluation/
│   ├── evaluate.py      # 테스트셋 mAP 평가
│   └── predict_folder.py# 폴더 단위 추론 (결과 이미지/라벨 저장)
├── results/             # 실험별 학습 로그(csv), 그래프, 최종 학습 설정
├── models/              # 최종 모델 위치 (Releases에서 다운로드)
├── data/                # 데이터셋 위치 (저장소 미포함)
└── config.py            # 클래스, 경로, 추론 기본값
```

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt  # GPU 사용 시 CUDA 버전에 맞는 PyTorch 별도 설치
# 최종 모델: Releases에서 best.pt를 받아 models/ 폴더에 저장
```

**웹앱**

```bash
python app/run.py
# http://127.0.0.1:5002  (models/best.pt 자동 로드)
```

**데이터 구축 → 학습 → 평가**

```bash
# 1) 1차 모델로 새 이미지 자동 라벨링 후 정리
python data_tools/auto_label.py --model models/best.pt --src data/raw --dst data/auto_labels
python data_tools/clean_labels.py --root data/auto_labels --apply

# 2) 수동 라벨 + 자동 라벨 병합, train/val 분할
python data_tools/build_dataset.py --src data/manual data/auto_labels --dst data/dataset

# 3) 학습 (최종 모델 설정)
python training/train.py --model yolo26m.pt --imgsz 768 --batch 5 --epochs 100 --name yolo26m_img768

# 4) 테스트셋 평가 (data/test/test.json + 이미지)
python evaluation/evaluate.py --model models/best.pt --test-dir data/test
```

## 데이터

학습 이미지(8,276장)는 웹에서 수집한 이미지라 저장소에 포함하지 않았습니다.
`data/dataset/`에 YOLO 형식(`images/{train,val}`, `labels/{train,val}`)으로 두면 바로 학습할 수 있습니다.

## 기술 스택

Python · Ultralytics YOLO (YOLOv8, YOLO26) · PyTorch · Flask · OpenCV

## 라이선스

[AGPL-3.0](LICENSE) — [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)(AGPL-3.0)를 사용하므로 같은 라이선스를 따릅니다. 최종 모델(`best.pt`)도 YOLO 사전학습 가중치를 기반으로 학습했으므로 동일하게 적용됩니다.
학습 이미지는 웹에서 수집한 것으로 저작권이 각 원작자에게 있어 저장소에 포함하지 않았습니다.
