// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../types';

import {createTensorShapeVariables, IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface ConcatAttributes extends AttributeWithCacheKey {
  readonly axis: number;
}

const validateInputs = (inputs: readonly TensorView[], referenceIndex: number): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('too few inputs');
  }

  const inputType = inputs[referenceIndex].dataType;
  const inputDimensionality = inputs[referenceIndex].dims.length;
  const referenceInput = inputs[referenceIndex];
  for (const input of inputs) {
    // make sure types of all inputs match
    if (input.dataType !== inputType) {
      throw new Error('input tensors should be one type');
    }

    // make sure the dimensionality of all inputs are the same
    if (input.dims.length !== inputDimensionality && ShapeUtil.size(input.dims) > 0 &&
        ShapeUtil.size(referenceInput.dims) > 0) {
      throw new Error('input tensors should have the same shape');
    }
  }
};

const calculateInputIndexImpl = (numberOfTensors: number, sizeInConcatAxisStr: string): string => `
  fn calculateInputIndex(index: u32) -> u32 {
    let sizeInConcatAxis = array<u32, ${numberOfTensors}u>(${sizeInConcatAxisStr});
    for (var i: u32 = 0u; i < ${numberOfTensors}; i += 1u ) {
      if (index < sizeInConcatAxis[i]) {
        return i;
      }
    }
    return ${numberOfTensors}u;
  }`;

const assignOutputData = (inputs: readonly IndicesHelper[], output: IndicesHelper) => {
  const numberOfTensors = inputs.length;

  const codeLines: string[] = [];
  for (let i = 0; i < numberOfTensors; ++i) {
    const returnSnippet = output.setByOffset('global_idx', inputs[i].getByIndices('indices'));
    if (numberOfTensors === 1) {
      codeLines.push(returnSnippet);
    } else if (i === 0) {
      codeLines.push(`if (inputIndex == ${i}u) { ${returnSnippet} }`);
    } else if (i === numberOfTensors - 1) {
      codeLines.push(`else { ${returnSnippet} }`);
    } else {
      codeLines.push(`else if (inputIndex == ${i}) { ${returnSnippet} }`);
    }
  }
  return codeLines.join('\n');
};

const computeReferenceIndex = (inputs: readonly TensorView[]): number => {
  // find a none zero tensor to determine the output shape
  let referenceIndex = 0;
  for (let j = 0; j < inputs.length; j++) {
    const size = ShapeUtil.size(inputs[j].dims);
    if (size > 0) {
      referenceIndex = j;
      break;
    }
  }
  return referenceIndex;
};

const computeOutputShape = (inputs: readonly TensorView[], axis: number, referenceIndex: number): number[] => {
  const inputShape = inputs[referenceIndex].dims.slice();
  if (axis >= inputShape.length || axis < (-1 * inputShape.length)) {
    throw new Error('axis specified for concat doesn\'t match input dimensionality');
  }
  // ensure all of the non-concatenated axes match each other
  // calculate the shape of the output tensor while we do that
  const outputShape = inputShape.slice(0);
  for (let i = 0; i < inputs.length; i++) {
    if (i === referenceIndex) {
      continue;
    }
    const dataNShape = inputs[i].dims.slice();
    for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
      // add to the placeholder for computing output shape
      if (axisIndex === axis) {
        outputShape[axis] += dataNShape[axisIndex];
      }
      // ensure all non-cancatenated axes match each other
      else if (inputShape[axisIndex] !== dataNShape[axisIndex] && ShapeUtil.size(dataNShape) > 0) {
        throw new Error('non concat dimensions must match');
      }
    }
  }
  return outputShape;
};

const createConcatProgramInfo = (inputs: readonly TensorView[], axis: number, outputShape: number[]): ProgramInfo => {
  const outputSize = ShapeUtil.size(outputShape);

  const sizeInConcatAxis = new Array<number>(inputs.length);
  const inputVars = new Array<IndicesHelper>(inputs.length);
  const dataType = inputs[0].dataType;

  let previousSum = 0;
  const inputDependencies: ProgramInputTensorInfoDependency[] = [];
  const inputRanks = [];
  const programUniforms: ProgramUniform[] = [{type: DataType.uint32, data: outputSize}];
  for (let i = 0; i < inputs.length; ++i) {
    previousSum += inputs[i].dims[axis];
    sizeInConcatAxis[i] = previousSum;
    inputRanks.push(inputs[i].dims.length);
    inputVars[i] = inputVariable(`input${i}`, dataType, inputRanks[i]);
    inputDependencies.push('rank');
    programUniforms.push({type: DataType.uint32, data: sizeInConcatAxis[i]});
  }
  for (let i = 0; i < inputs.length; ++i) {
    programUniforms.push(...createTensorShapeVariables(inputs[i].dims));
  }
  programUniforms.push(...createTensorShapeVariables(outputShape));

  const output = outputVariable('output', dataType, outputShape.length);
  const indicesAxis = output.indicesGet('indices', axis);
  const sizeInConcatAxisStr =
      Array.from(Array(sizeInConcatAxis.length).keys()).map(i => `uniforms.sizeInConcatAxis${i}`).join(',');
  const getShaderSource = (shaderHelper: ShaderHelper) => `

  ${(() => {
    shaderHelper.registerUniform('outputSize', 'u32');
    for (let i = 0; i < inputs.length; i++) {
      shaderHelper.registerUniform(`sizeInConcatAxis${i}`, 'u32');
    }
    return shaderHelper.declareVariables(...inputVars, output);
  })()}

  ${calculateInputIndexImpl(sizeInConcatAxis.length, sizeInConcatAxisStr)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.outputSize')}

    var indices = ${output.offsetToIndices('global_idx')};

    let inputIndex = calculateInputIndex(${indicesAxis});
    if (inputIndex < ${inputs.length}u) {
      if (inputIndex != 0u) {
        let sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}u>(${sizeInConcatAxisStr});
        ${indicesAxis} -= sizeInConcatAxis[inputIndex - 1u];
      }

      ${assignOutputData(inputVars, output)}
    } else {
      ${output.setByOffset('global_idx', '0')}
    }
  }`;

  return {
    name: 'Concat',
    shaderCache: {hint: `${axis}`, inputDependencies},
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
      programUniforms,
    }),
    getShaderSource,
  };
};

export const concat = (context: ComputeContext, attributes: ConcatAttributes): void => {
  const referenceIndex = computeReferenceIndex(context.inputs);
  validateInputs(context.inputs, referenceIndex);
  const axis = attributes.axis;
  const inputShape = context.inputs[referenceIndex].dims;
  const adjustedAxis = (attributes.axis < 0) ? inputShape.length + axis : axis;
  const outputShape = computeOutputShape(context.inputs, adjustedAxis, referenceIndex);
  // 0 length tensors are valid for concat, remove them
  const nonEmptyInputs = context.inputs.filter(input => ShapeUtil.size(input.dims) > 0);
  if (nonEmptyInputs.length > 0) {
    context.compute(createConcatProgramInfo(nonEmptyInputs, adjustedAxis, outputShape), {inputs: nonEmptyInputs});
  } else {
    context.output(0, outputShape);
  }
};

export const parseConcatAttributes = (attributes: Record<string, unknown>): ConcatAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis as number});
