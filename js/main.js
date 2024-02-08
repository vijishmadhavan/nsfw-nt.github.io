// Assuming OpenCV and face-api.js are already included in your HTML file

const labels = [
  "FEMALE_GENITALIA_COVERED",
  "FACE_FEMALE",
  "BUTTOCKS_EXPOSED",
  "FEMALE_BREAST_EXPOSED",
  "FEMALE_GENITALIA_EXPOSED",
  "MALE_BREAST_EXPOSED",
  "ANUS_EXPOSED",
  "FEET_EXPOSED",
  "BELLY_COVERED",
  "FEET_COVERED",
  "ARMPITS_COVERED",
  "ARMPITS_EXPOSED",
  "FACE_MALE",
  "BELLY_EXPOSED",
  "MALE_GENITALIA_EXPOSED",
  "ANUS_COVERED",
  "FEMALE_BREAST_COVERED",
  "BUTTOCKS_COVERED"
];

const useState = (defaultValue) => {
  let value = defaultValue;
  const getValue = () => value;
  const setValue = (newValue) => (value = newValue);
  return [getValue, setValue];
};

const numClass = labels.length;
const [session, setSession] = useState(null);
let mySession;

const modelName = "yolov8n.onnx";
const modelInputShape = [1, 3, 320, 320];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.2;

const handleImage = async (imageUrl) => {
  const image = new Image();
  image.crossOrigin = "anonymous";

  image.onload = async () => {
    console.log("Image loaded successfully");
    document.getElementById('displayImage').src = imageUrl;  // Display the image in the browser

    // Detect age and gender first
    const { detections } = await detectAgeFromImage(imageUrl);
    
    // Gather all detected ages
    const detectedAges = detections.map(detection => Math.round(detection.age));
    
    // Get the minimum detected age
    const minAge = Math.min(...detectedAges);
    
    // Pass the minimum detected age to detectImage
    detectImage(image, mySession, topk, iouThreshold, scoreThreshold, modelInputShape, minAge);
  };

  image.onerror = (error) => {
    console.error("Error loading the image: ", error);
  };

  image.src = imageUrl;
};

cv["onRuntimeInitialized"] = async () => {
  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri('models'),
      faceapi.nets.ageGenderNet.loadFromUri('models')
    ]);

    const [yolov8, nms] = await Promise.all([
      ort.InferenceSession.create(`model/${modelName}`),
      ort.InferenceSession.create(`model/nms-yolov8.onnx`),
    ]);
    mySession = setSession({ net: yolov8, nms: nms });

    console.log("YOLOv8 model and NMS model loaded successfully");

    handleImage("https://hotpotmedia.s3.us-east-2.amazonaws.com/8-yIhz1VJ5AkPkTIP.png");
  } catch (error) {
    console.error("Error loading models:", error);
  }
};

const detectImage = async (image, session, topk, iouThreshold, scoreThreshold, inputShape, detectedAge) => {
  console.log("detectImage function called");

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new ort.Tensor("float32", input.data32F, inputShape);
  const config = new ort.Tensor("float32", new Float32Array([topk, iouThreshold, scoreThreshold]));

  const { output0 } = await session.net.run({ images: tensor });
  const { selected } = await session.nms.run({ detection: output0, config: config });

  // Checking if the selected tensor has data
  if (!selected || !selected.data || selected.data.length === 0) {
    console.error("No data in selected tensor");
    return;
  }

  let labelsOfInterest = [
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED", 
    "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED"
  ];

  if (detectedAge < 15) {
    // If age is less than 15, include additional covered classes
    labelsOfInterest = labelsOfInterest.concat(["FEMALE_GENITALIA_COVERED", "BUTTOCKS_COVERED", "FEMALE_BREAST_COVERED", "ANUS_COVERED"]);
  }

  for (let idx = 0; idx < selected.dims[1]; idx++) {
    // Safely accessing the data with checks
    const startIdx = idx * selected.dims[2];
    const endIdx = startIdx + selected.dims[2];
    if (endIdx > selected.data.length) {
      console.error("Index out of bounds");
      continue;
    }

    const data = selected.data.slice(startIdx, endIdx);
    const score = Math.max(...data.slice(4));
    const labelIndex = data.indexOf(score) - 4;

    if (labelIndex < 0 || labelIndex >= labels.length) {
      console.error("Invalid label index");
      continue;
    }

    const detectedClass = labels[labelIndex];
    if (labelsOfInterest.includes(detectedClass)) {
      console.log(`Detected Class: ${detectedClass}, Probability: ${score * 100}%`);
    }
  }

  input.delete();
};

const preprocessing = (source, modelWidth, modelHeight) => {
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

// Add the detectAgeFromImage function here
async function detectAgeFromImage(imageUrl) {
  const img = new Image();
  img.crossOrigin = "anonymous"; // This is important for loading images from external URLs

  return new Promise((resolve, reject) => {
    img.onload = async () => {
      // Now that the image is loaded, we can detect faces and predict age/gender
      const detections = await faceapi.detectAllFaces(img).withAgeAndGender();
      // Log the results
      detections.forEach((detection, index) => {
        console.log(`Face ${index + 1}: Age - ${Math.round(detection.age)}, Gender - ${detection.gender}`);
      });
      resolve({ detections });
    };
    img.onerror = (error) => {
      console.error("Error loading the image: ", error);
      reject(error);
    };
    img.src = imageUrl;
  });
}


