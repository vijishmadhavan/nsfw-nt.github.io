const AllLabels = [
    'FEMALE_GENITALIA_COVERED',
    'FACE_FEMALE',
    'BUTTOCKS_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_BREAST_EXPOSED',
    'ANUS_EXPOSED',
    'FEET_EXPOSED',
    'BELLY_COVERED',
    'FEET_COVERED',
    'ARMPITS_COVERED',
    'ARMPITS_EXPOSED',
    'FACE_MALE',
    'BELLY_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'ANUS_COVERED',
    'FEMALE_BREAST_COVERED',
    'BUTTOCKS_COVERED'
];

const NsfwLabels = [
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'BUTTOCKS_EXPOSED',
    'BUTTOCKS_COVERED',
    'ANUS_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
];

const NsfwScore = 0.20;
let myYolo = null;
let myNms = null;
let faceapiInitialized = false;

// NSFW model data
const modelName = 'yolov8n.onnx';
const modelInputShape = [1, 3, 320, 320];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.2;

async function isNsfw(imageUrl) {
    if (!faceapiInitialized) {
        await initializeFaceAPI();
    }
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';

        image.onload = async () => {
            const tempCanvas = document.createElement('canvas');
            const tempContext = tempCanvas.getContext('2d');
            tempCanvas.width = image.width;
            tempCanvas.height = image.height;
            tempContext.drawImage(image, 0, 0, tempCanvas.width, tempCanvas.height);

            try {
                const result = await detectNsfw(tempCanvas, topk, iouThreshold, scoreThreshold, modelInputShape);
                resolve(result);
            } catch (error) {
                reject(error);
            }
        };

        image.onerror = (error) => {
            console.error('Error loading image: ', error);
            reject(error);
        };

        image.src = imageUrl;
    });
};

async function detectNsfw(image, topk, iouThreshold, scoreThreshold, inputShape) {
    const detectedAge = await detectAgeFromImage(image);
    console.log('Detected age:', detectedAge);

    if (detectedAge < 21) {
        // Age less than 21, additional covered classes are included
        NsfwLabels.push('FEMALE_GENITALIA_COVERED', 'BUTTOCKS_COVERED', 'FEMALE_BREAST_COVERED', 'ANUS_COVERED');
    }

    const [modelWidth, modelHeight] = inputShape.slice(2);
    const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

    const tensor = new ort.Tensor('float32', input.data32F, inputShape);
    const config = new ort.Tensor('float32', new Float32Array([topk, iouThreshold, scoreThreshold]));

    const { output0 } = await myYolo.run({ images: tensor });
    const { selected } = await myNms.run({ detection: output0, config: config });

    let foundNsfw = false;

    for (let idx = 0; idx < selected.dims[1]; idx++) {
        const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]);
        const box = data.slice(0, 4);
        const scores = data.slice(4);
        const score = Math.max(...scores);
        const label = scores.indexOf(score);

        const [x, y, w, h] = [
            (box[0] - 0.5 * box[2]) * xRatio,
            (box[1] - 0.5 * box[3]) * yRatio,
            box[2] * xRatio,
            box[3] * yRatio,
        ];

        const detectedClass = AllLabels[label];
        if (NsfwLabels.includes(detectedClass)) {
            console.log(`Detected class: ${detectedClass}, Score: ${score}`);
            if (score > NsfwScore) {
                foundNsfw = true;
                break;
            }
        }
    }

    input.delete();

    return foundNsfw;
};

function preprocessing(source, modelWidth, modelHeight) {
    const mat = cv.imread(source);
    const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);

    const maxSize = Math.max(matC3.rows, matC3.cols);
    const xPad = maxSize - matC3.cols,
        xRatio = maxSize / matC3.cols;
    const yPad = maxSize - matC3.rows,
        yRatio = maxSize / matC3.rows;
    const matPad = new cv.Mat();
    cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT);

    const input = cv.blobFromImage(
        matPad,
        1 / 255.0,
        new cv.Size(modelWidth, modelHeight),
        new cv.Scalar(0, 0, 0),
        true,
        false
    );

    mat.delete();
    matC3.delete();
    matPad.delete();

    return [input, xRatio, yRatio];
};

async function initializeFaceAPI() {
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('models'),
        faceapi.nets.ageGenderNet.loadFromUri('models')
    ]);
    faceapiInitialized = true;
}

async function detectAgeFromImage(image) {
    const detections = await faceapi.detectAllFaces(image).withAgeAndGender();
    if (detections.length === 0) {
        console.log('No face detected');
        return null;
    }
    const ages = detections.map(detection => detection.age);
    const averageAge = ages.reduce((total, age) => total + age, 0) / ages.length;
    return Math.round(averageAge);
}

cv['onRuntimeInitialized'] = async () => {
    try {
        const [yolov8, nms] = await Promise.all([
            ort.InferenceSession.create(`nsfw/${modelName}`),
            ort.InferenceSession.create(`nsfw/nms-yolov8.onnx`),
        ]);

        myYolo = yolov8;
        myNms = nms;

    } catch (error) {
        console.error('Error loading models: ', error);
    }
};


