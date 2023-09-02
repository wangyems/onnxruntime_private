// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface EinsumAttributes extends AttributeWithCacheKey {
  readonly equation: string;
}
// The equation attribute value is a string which consists of left hand side (LHS) and optionally right hand side (RHS)
// separated by '->'. Ex. "ij,jk -> ik" expresses matrix multiplication
//     "ij->ji" expresses matrix transpose
//      "ii->i" diagonal elements of a square matrix
// LHS consists of a sequence of terms separated by commas. Each term corresponds to an input variable.
// Each symbol corresponds to a dimension in the input variable. The symbol can be either a letter, 'a' to 'z' or 'A' to
// 'Z' or '...' to represent arbitrary dimensions.

const symbolPattern =
    '[a-zA-Z]|\\.\\.\\.';  // The pattern each symbol in each term in the symbolic equation should match
const termPattern = '(' + symbolPattern + ')+';   // The pattern each term in the symbolic equation should match
const termPatternOnly = '^' + termPattern + '$';  // The patterns only matchs a term begin to end.
const lhsPattern = '(' + termPattern + ',)*' + termPattern;  // The pattern the LHS should match
const lhsPatternOnly = '^' + lhsPattern + '$';               // The patterns only matchs a LHS begin to end.

class EinsumTerm {
  constructor(
      inputIndex = -1, symbols: string[] = [], symbolToIndices: Map<string, number[]> = new Map<string, number[]>()) {
    this.symbols = symbols;
    this.symbolToIndices = symbolToIndices;
    this.inputIndex = inputIndex;
  }
  symbols: string[];                       // All symbols in the term
  symbolToIndices: Map<string, number[]>;  // Map from symbol to dimensions of the input corresponding to the term
  inputIndex = -1;                         // -1 for output and 0, 1, 2, ... for inputs
}

class EinsumEquation {
  constructor(lhs: EinsumTerm[] = [], rhs: EinsumTerm = new EinsumTerm()) {
    this.lhs = lhs;
    this.rhs = rhs;
    this.symbolToDimValue = new Map<string, number>();
    this.symbolToCount = new Map<string, number>();
    this.symbolToInputIndices = new Map<string, number[]>();
  }
  symbolToDimValue: Map<string, number>;        // All symbols in the equation
  symbolToCount: Map<string, number>;           // Count how many times a symbol occoured in the equation on LHS.
  symbolToInputIndices: Map<string, number[]>;  // map to array of LSH terms the symbol is used.
  hasEllipsis = false;                          // The equation has ellipsis or not
  ellipsisDims: number[] = [];                  // The dimensions of the equation ellipsis corresponds to.
  lhs: EinsumTerm[];
  rhs: EinsumTerm;
  outputDims: number[] = [];
}

const createEinsumProgramMetadata = (inputCount: number, cacheHint: string): ProgramMetadata =>
    ({name: 'Einsum', inputTypes: Array(inputCount).fill(GpuDataType.default), cacheHint});

const addSymbol =
    (einsumEquation: EinsumEquation, einsumTerm: EinsumTerm, index: number, symbol: string, dimValue: number,
     inputIndex: number): void => {
      einsumTerm.symbols.push(symbol);
      if (einsumTerm.symbolToIndices.has(symbol)) {
        einsumTerm.symbolToIndices.get(symbol)!.push(index);
      } else {
        einsumTerm.symbolToIndices.set(symbol, [index]);
      }
      if (einsumEquation.symbolToDimValue.has(symbol) && einsumEquation.symbolToDimValue.get(symbol) !== 1) {
        if (einsumEquation.symbolToDimValue.get(symbol) !== dimValue) {
          throw new Error('Dimension mismatch');
        }
      } else {
        einsumEquation.symbolToDimValue.set(symbol, dimValue);
      }
      if (einsumEquation.symbolToInputIndices.has(symbol)) {
        einsumEquation.symbolToInputIndices.get(symbol)!.push(inputIndex);
      } else {
        einsumEquation.symbolToInputIndices.set(symbol, [inputIndex]);
      }
    };

// Process one input/output term
const processTerm =
    (term: string, isInput: boolean, dims: readonly number[], einsumEquation: EinsumEquation, index = -1) => {
      const rank = dims.length;
      let ellipsis = false;
      let ellipsisDims = [];
      let nextDim = 0;
      // For output empty string is allowed because the output may be reduced to a scalar value
      if (!term.match(RegExp(termPatternOnly)) && (!isInput && term !== '')) {
        throw new Error('Invalid LHS term');
      }
      const indexSymbols = term.match(RegExp(symbolPattern, 'g'));
      const einsumTerm = new EinsumTerm(index);
      // symbol can be either a lettre, 'a' to 'z' or 'A' to 'Z', or '...'
      indexSymbols?.forEach((symbol, i) => {
        if (symbol === '...') {
          if (ellipsis) {
            throw new Error('Only one ellipsis is allowed per input term');
          }
          ellipsis = true;
          const ellipsisDimLength = rank - indexSymbols.length + 1;
          if (ellipsisDimLength < 0) {
            throw new Error('Ellipsis out of bounds');
          }
          ellipsisDims = dims.slice(nextDim, nextDim + ellipsisDimLength);
          if (einsumEquation.hasEllipsis) {
            if (einsumEquation.ellipsisDims.length !== ellipsisDims.length ||
                einsumEquation.ellipsisDims.toString() !== ellipsisDims.toString()) {
              throw new Error('Ellipsis dimensions mismatch');
            }
          } else if (isInput) {
            einsumEquation.hasEllipsis = true;
            einsumEquation.ellipsisDims = ellipsisDims;
          } else {
            throw new Error('Ellipsis must be specified in the LHS');
          }
          // Add '0', '1', '2', '3', '4', etc to represent ellipsis dimensions to avoid special handling
          for (let j = 0; j < ellipsisDims.length; j++) {
            const symbol = String.fromCharCode('0'.charCodeAt(0) + i);
            addSymbol(einsumEquation, einsumTerm, i + j, symbol, dims[nextDim++], index);
          }
        } else {
          addSymbol(einsumEquation, einsumTerm, i, symbol, dims[nextDim++], index);
        }
      });
      return einsumTerm;
    };

const preprocessInputs = (inputs: readonly TensorView[], equation: string): EinsumEquation => {
  const einsumEquation = new EinsumEquation();
  // As rhs needs to be updated allow using let instead of const for both lhs and rhs.
  // eslint-disable-next-line prefer-const
  let [lhs, rhs] = equation.includes('->') ? equation.split('->', 2) : [equation, ''];
  if (!lhs.match(RegExp(lhsPatternOnly))) {
    throw new Error('Invalid LHS term');
  }
  const inputTerms = lhs.split(',');
  inputTerms.forEach((inputTerm, index) => {
    const dims = inputs[index].dims.slice();
    if (!inputTerm.match(RegExp(termPatternOnly))) {
      throw new Error('Invalid LHS term');
    }
    const einsumTerm = processTerm(inputTerm, true, dims, einsumEquation, index);
    einsumTerm.symbolToIndices.forEach((indices, symbol) => {
      einsumEquation.symbolToCount.set(
          symbol,
          (einsumEquation.symbolToCount.has(symbol) ? einsumEquation.symbolToCount.get(symbol)! : 0) + indices.length);
    });
    einsumEquation.lhs.push(einsumTerm);
  });

  // Initialize the RHS if not specified
  if (rhs === '') {
    // Construct RHS from LHS terms/symbols
    rhs += [...einsumEquation.symbolToCount.entries()]
               .filter(([sym, count]) => (count === 1 || sym === '...'))
               .map(([sym]) => sym)
               .join('');
  } else {
    if (!rhs.match(RegExp(termPattern))) {
      throw new Error('Invalid RHS');
    }
  }

  // Compute output dims
  const rhsSymbols = rhs.match(RegExp(symbolPattern, 'g'));
  rhsSymbols?.forEach((symbol) => {
    if (symbol === '...') {
      einsumEquation.outputDims = einsumEquation.outputDims.concat(einsumEquation.ellipsisDims);
    } else {
      if (!einsumEquation.symbolToDimValue.has(symbol)) {
        throw new Error('Invalid RHS symbol');
      }
      einsumEquation.outputDims.push(einsumEquation.symbolToDimValue.get(symbol)!);
    }
  });
  einsumEquation.rhs = processTerm(rhs, true, einsumEquation.outputDims, einsumEquation);
  return einsumEquation;
};

const createEinsumProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], einsumEquation: EinsumEquation): ProgramInfo => {
      const dataType = inputs[0].dataType;
      const inputVars = new Array<IndicesHelper>(inputs.length);
      for (let i = 0; i < inputs.length; ++i) {
        inputVars[i] = inputVariable(`input${i}`, dataType, inputs[i].dims);
      }
      const outputShape = einsumEquation.outputDims;
      const outputSize = ShapeUtil.size(outputShape);
      const output = outputVariable('output', dataType, outputShape);
      const idxCopy: string[] = [];
      const rhsSymbols = einsumEquation.rhs.symbols;
      const initProd = 'var prod = 1.0;';
      const initSum = 'var sum = 0.0;';
      const updateSum = 'sum += prod;';
      const reduceOpsSetIndices: string[] = [];
      const reduceOpsLoopHeaders: string[] = [];
      const reduceOpsLoopFooters: string[] = [];
      const reduceOpCompute: string[] = [];
      const isReduceOpsWithoutLoop = einsumEquation.symbolToCount.size === rhsSymbols.length;
      einsumEquation.symbolToCount.forEach((count, symbol) => {
        if (rhsSymbols.includes(symbol)) {
          const outputIndex = rhsSymbols.indexOf(symbol);
          einsumEquation.lhs.forEach((term, i) => {
            if (einsumEquation.symbolToInputIndices.has(symbol) &&
                einsumEquation.symbolToInputIndices.get(symbol)!.includes(i)) {
              term.symbolToIndices.get(symbol)!.forEach((index) => {
                idxCopy.push(`${
                    inputVars[i].indicesSet(
                        `input${i}Indices`, index, output.indicesGet('outputIndices', outputIndex))}`);
              });
            }
          });
        } else {
          einsumEquation.lhs.forEach((term, i) => {
            if (einsumEquation.symbolToInputIndices.has(symbol) &&
                einsumEquation.symbolToInputIndices.get(symbol)!.includes(i)) {
              term.symbolToIndices.get(symbol)!.forEach((index) => {
                reduceOpsSetIndices.push(`${inputVars[i].indicesSet(`input${i}Indices`, index, `${symbol}`)}`);
              });
              reduceOpCompute.push(`prod *= ${inputVars[i].getByIndices(`input${i}Indices`)};`);
            }
          });
          reduceOpsLoopHeaders.push(
              `for(var ${symbol}: u32 = 0; ${symbol} < ${einsumEquation.symbolToDimValue.get(symbol)}; ${symbol}++) {`);
          reduceOpsLoopFooters.push('}');
        }
      });
      const reduceOps = isReduceOpsWithoutLoop ?
          [
            ...idxCopy,
            `let sum = ${inputVars.map((inputVar, i) => inputVar.getByIndices(`input${i}Indices`)).join(' * ')};`
          ] :
          [
            ...idxCopy,
            initSum,
            ...reduceOpsLoopHeaders,
            ...reduceOpsSetIndices,
            initProd,
            ...reduceOpCompute,
            updateSum,
            ...reduceOpsLoopFooters,
          ];
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${shaderHelper.declareVariables(...inputVars, output)}

      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        var outputIndices = ${output.offsetToIndices('global_idx')};
        ${inputVars.map((inputVar, i) => `var input${i}Indices: ${inputVars[i].type.indices};`).join('\n')}
        ${reduceOps.join('\n')};
        ${output.setByOffset('global_idx', 'sum')};
      }`;
      return {
        ...metadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

const createEinsumProgramInfoLoader =
    (inputs: readonly TensorView[], einsumEquation: EinsumEquation, attributes: EinsumAttributes):
        ProgramInfoLoader => {
          const metadata = createEinsumProgramMetadata(inputs.length, attributes.cacheKey);
          return {...metadata, get: () => createEinsumProgramInfo(metadata, inputs, einsumEquation)};
        };

export const einsum = (context: ComputeContext, attributes: EinsumAttributes): void => {
  const einsumEquation = preprocessInputs(context.inputs, attributes.equation);
  context.compute(createEinsumProgramInfoLoader(context.inputs, einsumEquation, attributes));
};

export const parseEinsumAttributes = (attributes: Record<string, unknown>): EinsumAttributes => {
  const equation = (attributes.equation as string).replace(/\s+/g, '');
  return createAttributeWithCacheKey({equation});
};
