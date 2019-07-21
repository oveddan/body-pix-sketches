import {BodyPix, toMask} from '@tensorflow-models/body-pix';
import {OutputStride} from '@tensorflow-models/body-pix/dist/mobilenet';
import {BodyPixInput} from '@tensorflow-models/body-pix/dist/types';
import {getInputTensorDimensions, removePaddingAndResizeBack, resizeAndPadTo, scaleAndCropToInputTensorShape} from '@tensorflow-models/body-pix/dist/util';
import * as tf from '@tensorflow/tfjs-core';
import {expectNumbersClose} from '@tensorflow/tfjs-core/dist/test_util';


export const segmentationModelImageDimensions: [number, number] = [353, 257];

interface Padding {
  top: number;
  left: number;
  bottom: number;
  right: number;
}

export const cropAndScaleToInputSize =
    (tensor: tf.Tensor3D): tf.Tensor3D => {
      const [sourceHeight, sourceWidth] = tensor.shape;
      const [targetHeight, targetWidth] = segmentationModelImageDimensions;

      const scaleV = targetHeight / sourceHeight;

      const targetWidthThatMatchesAspect = Math.round(sourceWidth * scaleV);
      const cropW = targetWidthThatMatchesAspect - targetWidth;
      const widthStart = Math.floor(cropW / 2);
      const cropH = targetHeight


      console.log(
          'target, ', targetHeight, targetWidth, widthStart, cropH, cropW);

      // resize to match height.
      return tf.tidy(() => {
        const scaled =
            tensor.resizeBilinear([targetHeight, targetWidthThatMatchesAspect]);

        if (cropW > 0) {
          return scaled.slice([0, widthStart, 0], [cropH, cropW, 3]);
        } else {
          return scaled;
        }
      });
    }

const scaleToOutputSize =
    (tensor: tf.Tensor3D, {top, bottom, left, right}: Padding,
     [outputHeight, outputWidth]: [number, number]) => {
      return tf.tidy(() => {
        // remove padding that was added
        const [height, width] = tensor.shape;

        const cropH = height - (top + bottom);
        const cropW = width - (left + right);
        console.log(
            'padding', height, width, cropH, cropW, top, bottom, left, right);

        console.log('slicing to', top, left, cropH, cropW);
        const withPaddingRemoved =
            tf.slice3d(tensor, [top, left, 0], [cropH, cropW, tensor.shape[2]]);

        return withPaddingRemoved.resizeBilinear(
            [outputHeight, outputWidth], true);
      });
    };

export function estimatePersonSegmentation(
    model: BodyPix, input: BodyPixInput, outputStride: OutputStride = 16,
    [outputHeight, outputWidth]: [number, number],
    segmentationThreshold = 0.5): tf.Tensor2D {
  const [height, width] = getInputTensorDimensions(input);
  return tf.tidy(() => {
    const {
      resizedAndPadded,
      paddedBy: [[top, bottom], [left, right]],
    } = resizeAndPadTo(input, segmentationModelImageDimensions);

    const padding: Padding = {top, bottom, left, right};

    const segmentScores =
        model.predictForSegmentation(resizedAndPadded, outputStride);

    const inInputTensorSize = segmentScores.resizeBilinear(
        [resizedAndPadded.shape[0], resizedAndPadded.shape[1]], true);

    const scaled = removePaddingAndResizeBack(
        inInputTensorSize, [outputHeight, outputWidth],
        [[top, bottom], [left, right]]);

    return toMask(scaled.squeeze(), segmentationThreshold);
  });
};

const selectRed = (image: tf.Tensor3D): tf.Tensor2D => tf.tidy(
    () =>
        image.slice([0, 0, 0], [image.shape[0], image.shape[1], 1]).squeeze());

export const createBackgroundGrid = ([gridHeight, gridWidth]: [number, number],
                                     gridSpacing: number): tf.Tensor2D => {
  const canvas = document.createElement('canvas');
  canvas.width = gridWidth;
  canvas.height = gridHeight;

  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#ffffff'

  const gridCols = Math.floor(gridWidth / gridSpacing);

  for (let col = 0; col < gridCols; col++) {
    const y = 0;
    const x = col * gridSpacing;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, gridHeight);
    ctx.stroke();
  }

  const gridRows = Math.floor(gridHeight / gridSpacing);

  for (let row = 0; row < gridRows; row++) {
    const x = 0;
    const y = row * gridSpacing;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(gridWidth, y);
    ctx.stroke();
  }

  return tf.tidy(() => {
    return tf.browser.fromPixels(canvas, 1).squeeze().asType('float32').div(
               255) as tf.Tensor2D;
  });
};

const drawCircle =
    (ctx: CanvasRenderingContext2D, x: number, y: number, radius: number,
     opacity: number, fill: string) => {
      ctx.beginPath();
      ctx.globalAlpha = opacity;
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = fill;
      ctx.fill();
    }

export const createLightFilter = ([gridHeight, gridWidth]: [number, number],
                                  gridSpacing: number): tf.Tensor2D => {
  const canvas = document.createElement('canvas');
  canvas.width = gridWidth;
  canvas.height = gridHeight;

  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);


  const gridCols = Math.floor(gridWidth / gridSpacing);
  const gridRows = Math.floor(gridHeight / gridSpacing);

  for (let col = 0; col < gridCols; col++) {
    for (let row = 0; row < gridRows; row++) {
      const x = col * gridSpacing;
      const y = row * gridSpacing;
      const circleColor = '#ffffff';

      drawCircle(ctx, x, y, gridSpacing / 7, 1, circleColor);
      drawCircle(ctx, x, y, gridSpacing / 6, 0.9, circleColor);
      drawCircle(ctx, x, y, gridSpacing / 3, 0.7, circleColor);
      drawCircle(ctx, x, y, gridSpacing / 2, 0.6, circleColor);
    }
  }

  return tf.tidy(() => {
    const imageTexture =
        tf.browser.fromPixels(canvas, 1).squeeze().asType('float32').div(255) as
        tf.Tensor2D;

    return imageTexture.mul(imageTexture)
        .mul(imageTexture)
        .mul(imageTexture)
        .mul(imageTexture);
  });
};

// class GridProgram implements tf.webgl.GPGPUProgram {
//   variableNames = ['A'];
//   outputShape: number[];
//   userCode: string;

//   constructor(inputShape: number[]) {
//     // Element-wise operator generates the tensor whose shape is exactly
//     same
//     // with the input one.
//     this.outputShape = inputShape;
//     this.userCode = `
//       void main() {
//         float a = getAAtOutCoords();
//         float output = a * a;
//         setOutput(output);
//       }
//     `;
//   }
// }

// const program = new (GridProgram );
