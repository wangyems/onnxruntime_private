// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

export interface PadAttributes extends AttributeWithCacheKey {
  // 0-constant, 1-reflect, 2-edge, 3-wrap
  readonly mode: number;
  readonly value: number;
  readonly pads: number[];
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('Too few inputs');
  }
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Input type must be float.');
  }

  if (inputs.length >= 2) {
    let validPads = inputs[0].dims.length * 2 === inputs[1].dims[0];
    if (inputs.length === 4) {
      validPads = inputs[3].dims[0] * 2 === inputs[1].dims[0];
    }
    if (!validPads) {
      throw new Error('The pads should be a 1D tensor of shape [2 * input_rank] or [2 * num_axes].');
    }
  }
};

const getPadConstant =
    (outputDims: readonly number[], inputDims: readonly number[], inputStrides: readonly number[], pads: number[],
     dataType: string, constantValue: number): string => {
      const inputRank = inputDims.length;

      let block = '';
      for (let i = inputRank - 1; i >= 0; --i) {
        block += `
            k = i32(${outputDims.length < 2 ? 'indices' : `indices[${i}]`}) - ${pads[i]};
            if (k < 0) {
              break;
            }
            if (k >= ${inputDims[i]}) {
              break;
            }
            offset += k * ${inputStrides[i]};
        `;
      }

      return `
          value = ${dataType}(${constantValue});
          for (var i = 0; i < 1; i++) {
            var offset = 0;
            var k = 0;
            ${block}
            value = x[offset];
          }
      `;
    };

const getPadReflect =
    (outputDims: readonly number[], inputDims: readonly number[], inputStrides: readonly number[], pads: number[]):
        string => {
          const inputRank = inputDims.length;

          let block = '';
          for (let i = inputRank - 1; i >= 0; --i) {
            block += `
                k = i32(${outputDims.length < 2 ? 'indices' : `indices[${i}]`}) - ${pads[i]};
                if (k < 0) {
                  k = -k;
                }
                {
                  let _2n_1 = ${2 * (inputDims[i] - 1)};
                  k = i32(f32(k) % f32(_2n_1)) ;
                  if(k >= ${inputDims[i]}) {
                    k = _2n_1 - k;
                  }
                }
                offset += k * ${inputStrides[i]};
            `;
          }

          return `
              var offset = 0;
              var k = 0;
              ${block}
              value = x[offset];
          `;
        };

const getPadEdge =
    (outputDims: readonly number[], inputDims: readonly number[], inputStrides: readonly number[], pads: number[]):
        string => {
          const inputRank = inputDims.length;

          let block = '';
          for (let i = inputRank - 1; i >= 0; --i) {
            block += `
                k = i32(${outputDims.length < 2 ? 'indices' : `indices[${i}]`}) - ${pads[i]};
                if (k < 0) {
                  k = 0;
                }
                if (k >= ${inputDims[i]}) {
                  k = ${inputDims[i] - 1};
                }
                offset += k * ${inputStrides[i]};
            `;
          }

          return `
              var offset = 0;
              var k = 0;
              ${block}
              value = x[offset];
          `;
        };

const getPadWrap =
    (outputDims: readonly number[], inputDims: readonly number[], inputStrides: readonly number[], pads: number[]):
        string => {
          const inputRank = inputDims.length;

          let block = '';
          for (let i = inputRank - 1; i >= 0; --i) {
            block += `
                k = i32(${outputDims.length < 2 ? 'indices' : `indices[${i}]`}) - ${pads[i]};
                if (k < 0)  {
                  k += ${inputDims[i]};
                }
                if (k >= ${inputDims[i]}) {
                  k -= ${inputDims[i]};
                }
                offset += k * ${inputStrides[i]};
            `;
          }

          return `
              var offset = 0;
              var k = 0;
              ${block}
              value = x[offset];
          `;
        };

const getPadSnippet =
    (outputDims: readonly number[], inputDims: readonly number[], inputStrides: readonly number[],
     attributes: PadAttributes, dataType: string): string => {
      switch (attributes.mode) {
        case 0:
          return getPadConstant(outputDims, inputDims, inputStrides, attributes.pads, dataType, attributes.value);
        case 1:
          return getPadReflect(outputDims, inputDims, inputStrides, attributes.pads);
        case 2:
          return getPadEdge(outputDims, inputDims, inputStrides, attributes.pads);
        case 3:
          return getPadWrap(outputDims, inputDims, inputStrides, attributes.pads);
        default:
          throw new Error('Invalid mode');
      }
    };

const generatePadCode =
    (shaderHelper: ShaderHelper, inputDims: readonly number[], attributes: PadAttributes, dataType: string): string => {
      const outputDims = ShapeUtil.padShape(inputDims.slice(), attributes.pads);
      const outputSize = ShapeUtil.size(outputDims);
      const inputStrides = ShapeUtil.computeStrides(inputDims);
      const inputStridesRank = inputStrides.length;
      const padSnippet = getPadSnippet(outputDims, inputDims, inputStrides, attributes, dataType);

      const outputIndicesHelper = createIndicesHelper('output', outputDims);
      const xIndicesHelper = createIndicesHelper('x', inputDims);

      const padCode = `
          @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
          @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

          ${outputIndicesHelper.o2iImpl}
          ${xIndicesHelper.i2oImpl}

          ${shaderHelper.mainStart()}
            ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

            ${outputIndicesHelper.indicesVariableDeclaration('indices')}
            ${outputIndicesHelper.o2iCall('global_idx', 'indices')}
            ${outputIndicesHelper.indicesVariableDeclaration('xIndices')}
            ${outputIndicesHelper.o2iCall('global_idx', 'xIndices')}

            var offsets: array<u32, ${inputStridesRank}>;

            var value = ${dataType}(0);
            ${padSnippet}
            output[global_idx] = value;
          }`;
      return padCode;
    };

const createPadProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, attributes: PadAttributes): ProgramInfo => {
      const outputShape = ShapeUtil.padShape(inputs[0].dims.slice(), attributes.pads);
      return {
        ...metadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        getShaderSource: shaderHelper => generatePadCode(shaderHelper, inputs[0].dims, attributes, 'f32'),
        dispatchGroup: () => ({x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)})
      };
    };

const createPadAttributesFromInputs = (inputs: readonly TensorView[], attributes: PadAttributes): PadAttributes => {
  if (inputs.length > 1) {
    const bigInt64Pads = inputs[1].getBigInt64Array();
    const value = (inputs.length >= 3) ? inputs[2].getFloat32Array()[0] : 0.0;

    const inputRank = inputs[0].dims.length;
    const updatePads = new Int32Array(2 * inputRank).fill(0);
    if (inputs.length >= 4) {
      const axes = inputs[3].getBigInt64Array();
      for (let i = 0; i < axes.length; i++) {
        updatePads[Number(axes[i])] = Number(bigInt64Pads[i]);
        updatePads[Number(axes[i]) + inputRank] = Number(bigInt64Pads[i + axes.length]);
      }
    } else {
      bigInt64Pads.forEach((i, v) => updatePads[Number(i)] = (Number(v)));
    }

    const pads: number[] = [];
    updatePads.forEach(v => pads.push(v));

    return createAttributeWithCacheKey({mode: attributes.mode, value, pads});
  } else {
    return attributes;
  }
};

const createPadProgramInfoLoader = (inputs: readonly TensorView[], attributes: PadAttributes): ProgramInfoLoader => {
  const updatedAttributes = createPadAttributesFromInputs(inputs, attributes);
  const metadata:
      ProgramMetadata = {name: 'Pad', inputTypes: [GpuDataType.default], cacheHint: updatedAttributes.cacheKey};
  return {...metadata, get: () => createPadProgramInfo(inputs, metadata, updatedAttributes)};
};

export const pad = (context: ComputeContext, attributes: PadAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createPadProgramInfoLoader(context.inputs, attributes), {inputs: [0]});
};

export const parsePadAttributes = (attributes: Record<string, unknown>): PadAttributes => {
  const mode = attributes.mode as number;
  const value = attributes.value as number;
  const pads = attributes.pads as number[];
  return createAttributeWithCacheKey({mode, value, pads});
};
