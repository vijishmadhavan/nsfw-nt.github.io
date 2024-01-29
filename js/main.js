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

  image.onload = () => {
    console.log("Image loaded successfully");
    
    // Reference the main canvas, do not hide it
    const canvas = document.querySelector("canvas");
    // canvas.style.display = "none"; // This line is commented out to keep the canvas visible

    // Create a temporary canvas for detection (not displayed)
    const tempCanvas = document.createElement("canvas");
    const tempContext = tempCanvas.getContext("2d");
    tempCanvas.width = image.width;
    tempCanvas.height = image.height;
    tempContext.drawImage(image, 0, 0, tempCanvas.width, tempCanvas.height);

    // Now draw the image onto the main canvas
    const mainContext = canvas.getContext('2d');
    canvas.width = image.width; // Ensure the canvas size matches the image
    canvas.height = image.height;
    mainContext.drawImage(image, 0, 0, image.width, image.height);

    // Perform object detection using the temporary canvas
    detectImage(tempCanvas, null, mySession, topk, iouThreshold, scoreThreshold, modelInputShape);
  };

  image.onerror = (error) => {
    console.error("Error loading the image: ", error);
  };

  image.src = imageUrl;
};





cv["onRuntimeInitialized"] = async () => {
  try {
    const [yolov8, nms] = await Promise.all([
      ort.InferenceSession.create(`model/${modelName}`),
      ort.InferenceSession.create(`model/nms-yolov8.onnx`),
    ]);
    mySession = setSession({ net: yolov8, nms: nms });

    console.log("YOLOv8 model and NMS model loaded successfully");

    handleImage("https://hotpotmedia.s3.us-east-2.amazonaws.com/8-QWcxXaZTVRXFCUr.png");
  } catch (error) {
    console.error("Error loading models:", error);
  }
};

const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  console.log("detectImage function called");

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new ort.Tensor("float32", input.data32F, inputShape);
  const config = new ort.Tensor("float32", new Float32Array([topk, iouThreshold, scoreThreshold]));

  const { output0 } = await session.net.run({ images: tensor });
  const { selected } = await session.nms.run({ detection: output0, config: config });

  const boxes = [];
  const labelsOfInterest = [
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED", 
    "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED"
  ];

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

    boxes.push({
      label: label,
      probability: score,
      bounding: [x, y, w, h],
    });

    const detectedClass = labels[label];
    if (labelsOfInterest.includes(detectedClass)) {
      console.log(`Detected Class: ${detectedClass}, Probability: ${score * 100}%`);
    }
  }

  // Optionally, you can uncomment the next line to render boxes on the canvas
  // renderBoxes(canvas, boxes);

  input.delete();
};



const renderBoxes = (canvas, boxes) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const colors = new Colors();

  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const klass = labels[box.label];
    const color = colors.get(box.label);
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

    ctx.fillStyle = Colors.hexToRgba(color, 0.2);
    ctx.fillRect(x1, y1, width, height);

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(
      Math.min(ctx.canvas.width, ctx.canvas.height) / 200,
      2.5
    );
    ctx.strokeRect(x1, y1, width, height);

    ctx.fillStyle = color;
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10);
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    ctx.fillStyle = "#ffffff";
    ctx.fillText(
      klass + " - " + score + "%",
      x1 - 1,
      yText < 0 ? 1 : yText + 1
    );
  });
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

class Colors {
  constructor() {
    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];
  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? `rgba(${[
          parseInt(result[1], 16),
          parseInt(result[2], 16),
          parseInt(result[3], 16),
        ].join(", ")}, ${alpha})`
      : null;
  };
};


