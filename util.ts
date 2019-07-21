import * as bodyPix from '@tensorflow-models/body-pix';

type PartSegmentation = {
  data: Int32Array,
  width: number,
  height: number
};

export type BoundingBox = [number, number, number, number, number, number];

export const top = 0;
export const right = 1;
export const bottom = 2;
export const left = 3;
export const width = 4;
export const height = 5;

export function getPartBoundingBoxes(partSementation: PartSegmentation):
    BoundingBox[] {
  const {height: segmentationHeight, width: segmentationWidth, data} =
      partSementation;
  const boundingBoxes: BoundingBox[] = [];

  for (let i = 0; i < segmentationHeight * segmentationWidth; ++i) {
    // invert mask.  Invert the segmentatino mask.
    const partId = Math.round(data[i]);
    const j = i * 4;

    if (partId === -1) {
      continue;
    } else {
      const x = i % segmentationWidth;
      const y = Math.floor(i / segmentationWidth);

      if (!boundingBoxes[partId]) {
        boundingBoxes[partId] = [y, x, y, x, 0, 0];
      } else {
        const boundingBox = boundingBoxes[partId];

        boundingBox[top] = Math.min(boundingBox[top], y);
        boundingBox[right] = Math.max(boundingBox[right], x);
        boundingBox[bottom] = Math.max(boundingBox[bottom], y);
        boundingBox[left] = Math.min(boundingBox[left], x);
        boundingBox[height] = boundingBox[bottom] - boundingBox[top];
        boundingBox[width] = boundingBox[right] - boundingBox[left];
      }
    }
  }

  // console.log(boundingBoxes);
  return boundingBoxes;
}

/**
 * Shuffles array in place. ES6 version
 * @param {Array} a items An array containing the items.
 */
export function shuffle<T>(a: T[]): T[] {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function createOffScreenCanvas(): HTMLCanvasElement {
  const offScreenCanvas = document.createElement('canvas');
  return offScreenCanvas;
}

const offScreenCanvases: {[name: string]: HTMLCanvasElement} = {};

export function ensureOffscreenCanvasCreated(id: string): HTMLCanvasElement {
  if (!offScreenCanvases[id]) {
    offScreenCanvases[id] = createOffScreenCanvas();
  }
  return offScreenCanvases[id];
}


export function swapBox(
    a: BoundingBox, b: BoundingBox, ctx: CanvasRenderingContext2D) {
  // console.log(a[height], b[height], a[width], b[width]);

  if (a[height] * b[height] * a[width] * b[width] === 0) {
    return;
  }

  // console.log(a, b);

  const image = ctx.getImageData(a[left], a[top], a[width], a[height]);

  const swapImage = ctx.getImageData(b[left], b[top], b[width], b[height]);

  // console.log('swap');

  ctx.save();
  ctx.scale(b[width] / a[width], b[height] / a[height]);
  ctx.putImageData(swapImage, a[left], a[top]);
  ctx.restore();
  ctx.save();
  ctx.scale(a[width] / b[width], a[height] / b[height]);
  ctx.putImageData(image, b[left], b[top]);
  ctx.restore();
}

const rightHand = 21;
const leftHand = 23;
const leftFace = 0;
const rightFace = 1;

export function swapBoundingBoxes(
    partBoundingBoxes: BoundingBox[], ctx: CanvasRenderingContext2D) {
  for (let i = 0; i < 2; i++) {
    const randomPart = i === 0 ? leftHand : rightHand;

    const boundingBox = partBoundingBoxes[i];
    const randomBoundingBox = partBoundingBoxes[randomPart];

    if (boundingBox && randomBoundingBox) {
      swapBox(boundingBox, randomBoundingBox, ctx);
    }
  }
}

export function flip(ctx: CanvasRenderingContext2D) {
  const image = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-ctx.canvas.width, 0);

  ctx.putImageData(image, 0, 0);
  ctx.restore();
}

function drawBoundingBox(box: BoundingBox, ctx: CanvasRenderingContext2D) {
  ctx.rect(
      box[left], box[bottom], box[right] - box[left], box[top] - box[bottom]);
  ctx.stroke();
}

export function drawBoundingBoxes(
    partBoundingBoxes: BoundingBox[], ctx: CanvasRenderingContext2D) {
  partBoundingBoxes.forEach(box => {
    // draw bounding boxes
    drawBoundingBox(box, ctx);
  });
}

export async function loadImage(url: string) {
  const image = new Image();
  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = url;
  return promise;
}


function getFaceBox(partBoxes) {
  const top = 0;
  const right = 1;
  const bottom = 2;
  const left = 3;

  const leftFace = 0;
  const rightFace = 1;


  const leftFaceBox = partBoxes[leftFace];
  const rightFaceBox = partBoxes[rightFace];

  if (!leftFaceBox || !rightFaceBox) {
    return {top: 0, right: 0, bottom: 0, left: 0};
  }

  return {
    top: Math.min(leftFaceBox[top], rightFaceBox[top]),
    right: Math.max(leftFaceBox[right], rightFaceBox[right]),
    bottom: Math.max(leftFaceBox[bottom], rightFaceBox[bottom]),
    left: Math.min(leftFaceBox[left], rightFaceBox[left])
  };
}

export function drawOnFace(
    ctx: CanvasRenderingContext2D, partBoundingBoxes: BoundingBox[],
    faceImage: HTMLImageElement) {
  const faceBox = getFaceBox(partBoundingBoxes);

  const width = faceBox.right - faceBox.left;
  const height = faceBox.top - faceBox.bottom;

  ctx.drawImage(faceImage, faceBox.left, faceBox.bottom, width, height);
}
function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

// Here we use the camera label to determine if its the environment or
// user facing camera
function getFacingMode(cameraLabel) {
  if (!cameraLabel) {
    return 'user';
  }
  if (cameraLabel.toLowerCase().includes('back')) {
    return 'environment';
  } else {
    return 'user';
  }
}

async function getConstraints(cameraLabel) {
  let deviceId;
  let facingMode;

  if (cameraLabel) {
    // deviceId = await getDeviceIdForLabel(cameraLabel);
    // on mobile, use the facing mode based on the camera.
    facingMode = isMobile() ? getFacingMode(cameraLabel) : null;
  };
  return {facingMode};
}

/**
 * Loads a the camera to be used in the demo
 *
 */
export async function setupCamera(cameraLabel?: string) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const videoElement = document.getElementById('video') as HTMLVideoElement;

  const videoConstraints = await getConstraints(cameraLabel);

  const stream = await navigator.mediaDevices.getUserMedia(
      {'audio': false, 'video': videoConstraints});
  videoElement.srcObject = stream;

  return new Promise<HTMLVideoElement>((resolve) => {
    videoElement.onloadedmetadata = () => {
      videoElement.width = videoElement.videoWidth;
      videoElement.height = videoElement.videoHeight;
      resolve(videoElement);
    };
  });
}


export function scalesMismatch([height, width], [outputHeight, outputWidth]) {
  return height !== outputHeight || width !== outputWidth;
}

export function getFullScreenSize(
    [height, width, _], [screenHeight, screenWidth]): [number, number] {
  const aspect = width / height;
  const screenAspect = screenWidth / screenHeight;
  let scale;

  if (screenAspect > aspect) {
    // target width is greater than scaled width, scale to match height
    console.log('is wider');
    scale = screenHeight / height;
  } else {
    console.log('is taller');
    scale = screenWidth / width;
  }

  return [Math.round(height * scale), Math.round(width * scale)];
}
