import * as bodyPix from '@tensorflow-models/body-pix';
import {PartSegmentation} from '@tensorflow-models/body-pix/dist/types';
import * as tf from '@tensorflow/tfjs-core';

import {createBackgroundGrid, estimatePersonSegmentation} from './ops';
import {bottom, BoundingBox, drawBoundingBoxes, drawOnFace, ensureOffscreenCanvasCreated, getPartBoundingBoxes, height, left, loadImage, right, setupCamera, shuffle, swapBox, top, width} from './util';

type State = {
  video: HTMLVideoElement,
  net: bodyPix.BodyPix
}

const state: State = {
  video: null,
  net: null
};


export async function loadVideo(cameraLabel?: string) {
  try {
    state.video = await setupCamera(cameraLabel);
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  state.video.play();
}


// const outputWidth = 640;
// const outputHeight = 480;


const opacity = 0.02;
const feedbackStrength = 1 - opacity;

const getAndScaleLastFrame =
    (lastFrame: tf.Tensor2D,
     [height, width]: [number, number]): tf.Tensor2D => {
      if (!lastFrame) return tf.zeros([height, width]);

      if (lastFrame) {
        const [lastHeight, lastWidth] = lastFrame.shape;
        if (lastHeight === height && lastWidth === width) {
          return lastFrame;
        }
        return (lastFrame.expandDims(2) as tf.Tensor3D)
            .resizeBilinear([height, width])
            .squeeze();
      }
    }

const feedbackMask =
    (partSegmentation: tf.Tensor2D, lastFrame: tf.Tensor2D) => {
      return tf.tidy(() => {
        // const data = await partSegmentation.data();
        // console.log('data', data);

        const mask = partSegmentation.mul(opacity) as tf.Tensor2D;

        const lastFrameToUse = getAndScaleLastFrame(lastFrame, mask.shape);

        const frame =
            mask.add(lastFrameToUse.mul(feedbackStrength)) as tf.Tensor2D;

        return frame.clipByValue(0, 1);
      });
    };

const flipHorizontally = true;
const fullScreen = false;

const [gridHeight, gridWidth] = [480, 640];

const gridSpacing = 10;

const gridRows = 60;
const gridCols = 60;

const lightColor: [number, number, number] = [1, 0, .1];
const notLitColor: [number, number, number] = [0.2, 0.2, 0.2];

/**
 * Feeds an image to BodyPix to estimate segmentation -
 * this is where the magic happens. This function loops
 * with a requestAnimationFrame method.
 */
async function segmentBodyInRealTime() {
  const canvas = document.getElementById('output') as HTMLCanvasElement;
  // since images are being fed from a webcam

  let lastMask: tf.Tensor2D;

  const backgroundGrid = createBackgroundGrid(
      [gridHeight, gridWidth], gridSpacing, [gridCols, gridRows]);

  const lightColorTensor = tf.tensor3d(lightColor, [1, 1, 3]);
  const notLitTensor = tf.tensor3d(notLitColor, [1, 1, 3]);

  async function bodySegmentationFrame() {
    // if changing the model or the camera, wait a
    // second for it to complete then try again.

    // Scale an image down to a certain factor. Too
    // large of an image will slow down the GPU
    const outputStride = 8;

    const segmentationThreshold = 0.5;

    const input = await tf.browser.fromPixels(state.video);

    const outputWidth = fullScreen ? window.innerWidth : 640;
    const outputHeight = fullScreen ? window.innerHeight : 480;

    const mask = tf.tidy(() => {
      const partSegmentation = estimatePersonSegmentation(
          state.net, input, outputStride, [outputHeight, outputWidth],
          segmentationThreshold);

      const flippedSegmentation =
          flipHorizontally ? partSegmentation.reverse(1) : partSegmentation;

      const feedback = feedbackMask(flippedSegmentation, lastMask);

      return feedback;
    });

    lastMask = mask;

    const output = tf.tidy(() => {
      const scaledGrid = (backgroundGrid.expandDims(2) as tf.Tensor3D)
                             .resizeBilinear([outputHeight, outputWidth])
                             .squeeze();


      const maskedGrid = scaledGrid.mul(mask);

      const notLitGrid =
          scaledGrid.expandDims(2).mul(notLitTensor) as tf.Tensor3D;

      const rgbGrid =
          maskedGrid.expandDims(2).mul(lightColorTensor) as tf.Tensor3D;

      // return scaledGrid.mul(mask) as tf.Tensor2D;
      return notLitGrid.add(rgbGrid).clipByValue(0, 1) as tf.Tensor3D;
    });

    const ctx = canvas.getContext('2d');

    ctx.save();

    if (flipHorizontally) {
      ctx.scale(-1, 1);
      ctx.translate(-ctx.canvas.width, 0);
    }

    console.log('drawing');
    await tf.browser.toPixels(output, canvas);
    console.log('drew');

    output.dispose();
    input.dispose();

    ctx.restore();

    requestAnimationFrame(bodySegmentationFrame);
  }

  bodySegmentationFrame();
}

/**
 * Kicks off the demo.
 */
export async function bindPage() {
  // Load the BodyPix model weights with architecture
  // 0.75
  state.net = await bodyPix.load(0.50);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'inline-block';

  await loadVideo();

  // setupFPS();

  segmentBodyInRealTime();
}


navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
