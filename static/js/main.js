const modelName = document.getElementById("modelName");
const gpuStatus = document.getElementById("gpuStatus");
const modelFile = document.getElementById("modelFile");
const uploadModelBtn = document.getElementById("uploadModelBtn");
const modelMessage = document.getElementById("modelMessage");

async function loadModelStatus() {
    try {
        const response = await fetch("/model_status");
        const data = await response.json();

        if (data.loaded) {
            modelName.textContent = data.model_name;
        } else {
            modelName.textContent = "모델 없음";
        }

        if (data.cuda) {
            gpuStatus.textContent = `GPU 사용 가능: ${data.gpu}`;
        } else {
            gpuStatus.textContent = "GPU 사용 불가 / CPU 사용";
        }
    } catch (error) {
        modelName.textContent = "상태 확인 실패";
        gpuStatus.textContent = "확인 실패";
    }
}

uploadModelBtn.addEventListener("click", async () => {
    if (!modelFile.files.length) {
        modelMessage.textContent = "먼저 .pt 모델 파일을 선택하세요.";
        modelMessage.className = "message error";
        return;
    }

    const file = modelFile.files[0];

    if (!file.name.toLowerCase().endsWith(".pt")) {
        modelMessage.textContent = ".pt 파일만 업로드할 수 있습니다.";
        modelMessage.className = "message error";
        return;
    }

    const formData = new FormData();
    formData.append("model", file);

    uploadModelBtn.disabled = true;
    uploadModelBtn.textContent = "업로드 중...";
    modelMessage.textContent = "모델을 업로드하고 로드하는 중입니다.";
    modelMessage.className = "message";

    try {
        const response = await fetch("/upload_model", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            modelName.textContent = data.model_name;
            modelMessage.textContent = "모델 업로드 완료: " + data.model_name;
            modelMessage.className = "message success";
        } else {
            modelMessage.textContent = data.error;
            modelMessage.className = "message error";
        }
    } catch (error) {
        modelMessage.textContent = "업로드 중 오류가 발생했습니다.";
        modelMessage.className = "message error";
    }

    uploadModelBtn.disabled = false;
    uploadModelBtn.textContent = "모델 업로드";

    loadModelStatus();
});

loadModelStatus();


const imageFile = document.getElementById("imageFile");
const predictBtn = document.getElementById("predictBtn");
const predictMessage = document.getElementById("predictMessage");
const topClass = document.getElementById("topClass");
const topConfidence = document.getElementById("topConfidence");
const numDetections = document.getElementById("numDetections");
const resultArea = document.getElementById("resultArea");
const detectionList = document.getElementById("detectionList");

predictBtn.addEventListener("click", async () => {
    if (!imageFile.files.length) {
        predictMessage.textContent = "먼저 이미지를 선택하세요.";
        predictMessage.className = "message error";
        return;
    }

    const file = imageFile.files[0];

    const formData = new FormData();
    formData.append("image", file);

    predictBtn.disabled = true;
    predictBtn.textContent = "예측 중...";
    predictMessage.textContent = "이미지를 분석하는 중입니다.";
    predictMessage.className = "message";

    topClass.textContent = "-";
    topConfidence.textContent = "-";
    numDetections.textContent = "-";
    resultArea.innerHTML = `<div class="placeholder">예측 중...</div>`;
    detectionList.innerHTML = "";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!data.success) {
            predictMessage.textContent = data.error;
            predictMessage.className = "message error";
            resultArea.innerHTML = `<div class="placeholder">예측 실패</div>`;
            return;
        }

        predictMessage.textContent = `예측 완료 - 사용 모델: ${data.model_name}`;
        predictMessage.className = "message success";

        topClass.textContent = data.top_class;
        topConfidence.textContent = data.top_confidence;
        numDetections.textContent = data.num_detections;

        resultArea.innerHTML = `
            <img class="result-image" src="data:image/png;base64,${data.result_image}" alt="prediction result">
        `;

        if (data.detections.length > 0) {
            detectionList.innerHTML = "<h3>탐지 목록</h3>";

            data.detections.forEach((det, index) => {
                const item = document.createElement("div");
                item.className = "detection-item";
                item.innerHTML = `
                    <strong>${index + 1}. ${det.class}</strong>
                    <span>confidence: ${det.confidence}</span>
                    <span>bbox: [${det.bbox.join(", ")}]</span>
                `;
                detectionList.appendChild(item);
            });
        } else {
            detectionList.innerHTML = "<p>탐지된 벌이 없습니다.</p>";
        }

    } catch (error) {
        predictMessage.textContent = "예측 중 오류가 발생했습니다.";
        predictMessage.className = "message error";
        resultArea.innerHTML = `<div class="placeholder">예측 실패</div>`;
    }

    predictBtn.disabled = false;
    predictBtn.textContent = "예측 실행";
});

const evaluateBtn = document.getElementById("evaluateBtn");
const evaluateMessage = document.getElementById("evaluateMessage");
const map50 = document.getElementById("map50");
const map5095 = document.getElementById("map5095");
const precision = document.getElementById("precision");
const recall = document.getElementById("recall");

evaluateBtn.addEventListener("click", async () => {
    evaluateBtn.disabled = true;
    evaluateBtn.textContent = "mAP 계산 중...";
    evaluateMessage.textContent = "test.json과 test 이미지를 평가하는 중입니다. 시간이 걸릴 수 있습니다.";
    evaluateMessage.className = "message";

    map50.textContent = "-";
    map5095.textContent = "-";
    precision.textContent = "-";
    recall.textContent = "-";

    try {
        const response = await fetch("/evaluate", {
            method: "POST"
        });

        const data = await response.json();

        if (!data.success) {
            evaluateMessage.textContent = data.error;
            evaluateMessage.className = "message error";
            return;
        }

        map50.textContent = data.map50;
        map5095.textContent = data.map5095;
        precision.textContent = data.precision;
        recall.textContent = data.recall;

        evaluateMessage.textContent =
            `평가 완료 - 모델: ${data.model_name}, 이미지: ${data.images}장, 객체: ${data.annotations}개`;
        evaluateMessage.className = "message success";

    } catch (error) {
        evaluateMessage.textContent = "mAP 계산 중 오류가 발생했습니다.";
        evaluateMessage.className = "message error";
    }

    evaluateBtn.disabled = false;
    evaluateBtn.textContent = "test 데이터 mAP 계산";
});