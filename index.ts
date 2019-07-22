import * as bodyPix from '@tensorflow-models/body-pix';
import {PartSegmentation} from '@tensorflow-models/body-pix/dist/types';
import * as tf from '@tensorflow/tfjs-core';

import {createBackgroundGrid, createLightFilter, cropAndScaleToInputSize, drawMaskFilter, estimatePersonSegmentation, getPadToMatch, padToMatch} from './ops';
import {bottom, BoundingBox, drawBoundingBoxes, drawOnFace, ensureOffscreenCanvasCreated, getFullScreenSize, getPartBoundingBoxes, height, left, loadImage, right, scalesMismatch, setupCamera, shuffle, swapBox, top, width} from './util';

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


const opacity = 0.3;
const feedbackStrength = 1 - 0.05;

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


const gridSpacing = 13;

const notLitOpacity = 0.2;

const lightColor: [number, number, number] = [1, 0, .1];
const notLitColor: [number, number, number] =
    [notLitOpacity, notLitOpacity, notLitOpacity];

/**
 * Feeds an image to BodyPix to estimate segmentation -
 * this is where the magic happens. This function loops
 * with a requestAnimationFrame method.
 */
async function segmentBodyInRealTime() {
  const canvas = document.getElementById('output') as HTMLCanvasElement;
  // since images are being fed from a webcam

  let lastMask: tf.Tensor2D;
  const [gridHeight, gridWidth] = [state.video.height, state.video.width];

  let backgroundGrid: tf.Tensor2D;

  let lightFilter: tf.Tensor2D;
  let maskFilter: tf.Tensor2D;

  const lightColorTensor = tf.tensor3d(lightColor, [1, 1, 3]);
  const notLitTensor = tf.tensor3d(notLitColor, [1, 1, 3]);

  const imageCropTensor = tf.browser.fromPixels(
      document.getElementById('imageCrop') as HTMLImageElement);


  async function bodySegmentationFrame() {
    // if changing the model or the camera, wait a
    // second for it to complete then try again.

    // Scale an image down to a certain factor. Too
    // large of an image will slow down the GPU
    const outputStride = 8;

    const segmentationThreshold = 0.5;

    const input = await tf.browser.fromPixels(state.video);

    const [screenHeight, screenWidth] = [window.innerHeight, window.innerWidth];

    const [fullScreenHeight, fullScreenWidth] =
        getFullScreenSize(input.shape, [screenHeight, screenWidth]);

    const outputWidth = fullScreen ? fullScreenWidth : input.shape[1];
    const outputHeight = fullScreen ? fullScreenHeight : input.shape[0];

    console.log('target', fullScreenHeight, fullScreenWidth);

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

    if (!backgroundGrid ||
        scalesMismatch(backgroundGrid.shape, [outputHeight, outputWidth])) {
      if (backgroundGrid) backgroundGrid.dispose();
      backgroundGrid =
          createBackgroundGrid([outputHeight, outputWidth], gridSpacing);
    }

    if (!lightFilter ||
        scalesMismatch(lightFilter.shape, [outputHeight, outputWidth])) {
      if (lightFilter) lightFilter.dispose();
      lightFilter = createLightFilter([outputHeight, outputWidth], gridSpacing);
    }

    if (!maskFilter ||
        scalesMismatch(maskFilter.shape, [outputHeight, outputWidth])) {
      if (maskFilter) maskFilter.dispose();

      maskFilter = drawMaskFilter([outputHeight, outputWidth]);
    }

    const output = tf.tidy(() => {
      const scaledGrid = (backgroundGrid.expandDims(2) as tf.Tensor3D)
                             .resizeBilinear([outputHeight, outputWidth])
                             .squeeze();


      const maskedGrid = scaledGrid.mul(mask) as tf.Tensor2D;

      const notLitGrid =
          scaledGrid.expandDims(2).mul(notLitTensor) as tf.Tensor3D;


      // const scaledFilter = (lightFilter.expandDims(2) as tf.Tensor3D)
      //                          .resizeBilinear([outputHeight, outputWidth])
      //                          .squeeze();

      // console.log('not list grid', notLitGrid.shape, scaledFilter.shape);

      const lightGrid = maskedGrid.mul(lightFilter);


      const rgbGrid =
          lightGrid.expandDims(2).mul(lightColorTensor) as tf.Tensor3D;


      // console.log('not lit', rgbGrid.shape, notLitGrid.shape);
      // return scaledGrid.mul(mask) as tf.Tensor2D;
      return notLitGrid.add(rgbGrid).clipByValue(0, 1) as tf.Tensor3D;
    });

    console.log('masking')
    const croppedByMask = output.mul(maskFilter) as tf.Tensor3D;

    console.log('masked', output.dtype, croppedByMask.dtype);

    const padded = padToMatch(croppedByMask, [screenHeight, screenWidth]);

    console.log('padding', padded.dtype);

    const ctx = canvas.getContext('2d');

    ctx.save();

    if (flipHorizontally) {
      ctx.scale(-1, 1);
      ctx.translate(-ctx.canvas.width, 0);
    }

    // console.log('drawing');
    await tf.browser.toPixels(padded, canvas);
    // console.log('drew');

    output.dispose();
    croppedByMask.dispose();
    padded.dispose();
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
