// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {NUMBER_TYPES, OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';

export interface SliceAttributes {
  axes: number[];
  ends: number[];
  starts: number[];
}

export const slice: OperatorImplementation<SliceAttributes> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], attributes: SliceAttributes): Tensor[] => {
      validateInputs(inputs);
      const output = inferenceHandler.run(createSliceProgramInfo(inferenceHandler, inputs[0], attributes), inputs);
      return [output];
    };

export const parseSliceAttributes: OperatorInitialization<SliceAttributes> = (node: Graph.Node): SliceAttributes => {
  const starts = node.attributes.getInts('starts');
  const ends = node.attributes.getInts('ends');
  const axes = node.attributes.getInts('axes', []);
  return {starts, ends, axes};
};

const createSliceProgramInfo =
    (inferenceHandler: WebGLInferenceHandler, input: Tensor, attributes: SliceAttributes): ProgramInfo => {
      if (attributes.axes.length === 0) {
        attributes.axes = input.dims.slice(0).map((val, i) => i);
      }
      const axes = ShapeUtil.normalizeAxes(attributes.axes, input.dims.length);
      const starts = attributes.starts.map((start, i) => {
        if (start > input.dims[axes[i]] - 1) {
          return input.dims[axes[i]];
        }
        return ShapeUtil.normalizeAxis(start, input.dims[axes[i]]);
      });
      const ends = attributes.ends.map((end, i) => {
        if (end > input.dims[axes[i]] - 1) {
          return input.dims[axes[i]];
        }
        return ShapeUtil.normalizeAxis(end, input.dims[axes[i]]);
      });

      const outputShape = input.dims.slice();

      const sliceOps: string[] = [];
      for (let i = 0; i < axes.length; i++) {
        outputShape[axes[i]] = ends[i] - starts[i];
        if (starts[i] > 0) {
          sliceOps.push(`outputIdx[${axes[i]}] += ${starts[i]};`);
        }  // else { sliceOps.push(`outputIdx[${axes[i]}] += 0;`); }
      }

      const rank = outputShape.length;
      const shaderSource = `
      float process(int outputIdx[${rank}]) {
        ${sliceOps.join('\n      ')}
        return _A(outputIdx);
      }`;
      return {
        name: 'Slice',
        inputNames: ['A'],
        inputTypes: [TextureType.unpacked],
        output: {dims: outputShape, type: input.type, textureType: TextureType.unpacked},
        shaderSource
      };
    };

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Slice requires 1 input.');
  }
  if (NUMBER_TYPES.indexOf(inputs[0].type) === -1) {
    throw new Error('Invalid input type.');
  }
};

export const sliceV10 = (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] => {
  validateInputsV10(inputs);
  const output = inferenceHandler.run(createSliceProgramInfoV10(inferenceHandler, inputs), inputs);
  return [output];
};

const createSliceProgramInfoV10 = (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo => {
  if (!inferenceHandler.session.isInitializer(inputs[1].dataId) ||
      !inferenceHandler.session.isInitializer(inputs[2].dataId) ||
      (inputs.length >= 4 && !inferenceHandler.session.isInitializer(inputs[3].dataId)) ||
      (inputs.length >= 5 && !inferenceHandler.session.isInitializer(inputs[4].dataId))) {
    throw new Error('dynamic slice attributes are not allowed');
  }

  if (inputs.length >= 5 && inputs[4].integerData.some((i: number) => i !== 1)) {
    throw new Error('currently non-1 steps is not supported for Slice');
  }

  const starts = Array.from(inputs[1].integerData);
  const ends = Array.from(inputs[2].integerData);
  const axes = inputs.length >= 4 ? Array.from(inputs[3].integerData) : [];

  return createSliceProgramInfo(inferenceHandler, inputs[0], {starts, ends, axes});
};

const validateInputsV10 = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length < 3 || inputs.length > 5) {
    throw new Error('Invalid input shape.');
  }
  if (inputs[1].type !== 'int32' || inputs[1].dims.length !== 1) {
    throw new Error('Invalid input type.');
  }
  if (inputs[2].type !== 'int32' || inputs[2].dims.length !== 1) {
    throw new Error('Invalid input type.');
  }
  if (inputs.length >= 4 && (inputs[3].type !== 'int32' || inputs[3].dims.length !== 1)) {
    throw new Error('Invalid input type.');
  }
  if (inputs.length >= 5 && (inputs[4].type !== 'int32' || inputs[4].dims.length !== 1)) {
    throw new Error('Invalid input type.');
  }
};