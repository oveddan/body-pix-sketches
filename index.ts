import * as bodyPix from '@tensorflow-models/body-pix';
import {PartSegmentation} from '@tensorflow-models/body-pix/dist/types';
import * as tf from '@tensorflow/tfjs-core';
import * as datGui from 'dat.gui';

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

interface GuiState {
  appearanceSpeed: number, burnStrength: number, gridSpacing: number,
      notLitOpacity: number, lightColor: [number, number, number]
  cutIrregularEdge: boolean, fullScreen: boolean
}

const guiState: GuiState = {
  appearanceSpeed: 0.3,
  burnStrength: .95,
  gridSpacing: 19,
  notLitOpacity: 0.12,
  lightColor: [255, 0, 10],
  cutIrregularEdge: false,
  fullScreen: false
};

const createGui = () => {
  const gui = new datGui.GUI();
  gui.add(guiState, 'appearanceSpeed').min(.2).max(1);
  gui.add(guiState, 'burnStrength').min(0).max(.9);
  gui.add(guiState, 'gridSpacing').min(5).max(30).step(1);
  gui.add(guiState, 'notLitOpacity').min(0).max(.5);
  gui.addColor(guiState, 'lightColor');
  gui.add(guiState, 'cutIrregularEdge');
  gui.add(guiState, 'fullScreen');
};

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

        const mask =
            partSegmentation.mul(guiState.appearanceSpeed) as tf.Tensor2D;

        const lastFrameToUse = getAndScaleLastFrame(lastFrame, mask.shape);

        const frame =
            mask.add(lastFrameToUse.mul(guiState.burnStrength)) as tf.Tensor2D;

        return frame.clipByValue(0, 1);
      });
    };

const flipHorizontally = true;

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
  let backgroundGridSpacing: number;

  let lightFilter: tf.Tensor2D;
  let maskFilter: tf.Tensor2D;

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

    const outputWidth = guiState.fullScreen ? fullScreenWidth : input.shape[1];
    const outputHeight =
        guiState.fullScreen ? fullScreenHeight : input.shape[0];

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

    if (!backgroundGrid || backgroundGridSpacing !== guiState.gridSpacing ||
        scalesMismatch(backgroundGrid.shape, [outputHeight, outputWidth])) {
      if (backgroundGrid) backgroundGrid.dispose();
      backgroundGrid = createBackgroundGrid(
          [outputHeight, outputWidth], guiState.gridSpacing);
      backgroundGridSpacing = guiState.gridSpacing;

      if (lightFilter) lightFilter.dispose();
      lightFilter =
          createLightFilter([outputHeight, outputWidth], guiState.gridSpacing);
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

      const notLitColor = [
        guiState.notLitOpacity, guiState.notLitOpacity, guiState.notLitOpacity
      ];

      const notLitTensor = tf.tensor3d(notLitColor, [1, 1, 3]);

      const notLitGrid =
          scaledGrid.expandDims(2).mul(notLitTensor) as tf.Tensor3D;


      // const scaledFilter = (lightFilter.expandDims(2) as tf.Tensor3D)
      //                          .resizeBilinear([outputHeight, outputWidth])
      //                          .squeeze();

      // console.log('not list grid', notLitGrid.shape, scaledFilter.shape);

      const lightGrid = maskedGrid.mul(lightFilter);

      const lightColorTensor = tf.tensor3d(guiState.lightColor, [
                                   1, 1, 3
                                 ]).div(tf.scalar(255, 'float32'));


      const rgbGrid =
          lightGrid.expandDims(2).mul(lightColorTensor) as tf.Tensor3D;

      // console.log('not lit', rgbGrid.shape, notLitGrid.shape);
      // return scaledGrid.mul(mask) as tf.Tensor2D;
      return notLitGrid.add(rgbGrid).clipByValue(0, 1) as tf.Tensor3D;
    });

    const croppedByMask = guiState.cutIrregularEdge ?
        (output.mul(maskFilter) as tf.Tensor3D) :
        output;

    const padded = padToMatch(croppedByMask, [screenHeight, screenWidth]);

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
  createGui();

  segmentBodyInRealTime();
}


navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
