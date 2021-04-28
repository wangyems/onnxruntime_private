// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {onnx} from 'onnx-proto';

import {Graph} from './graph';
import {OpSet} from './opset';
import {LongUtil} from './util';

export class Model {
  // empty model
  constructor() {}

  load(buf: Uint8Array, graphInitializer?: Graph.Initializer): void {
    const modelProto = onnx.ModelProto.decode(buf);
    const irVersion = LongUtil.longToNumber(modelProto.irVersion);
    if (irVersion < 3) {
      throw new Error('only support ONNX model with IR_VERSION>=3');
    }

    this._opsets =
        modelProto.opsetImport.map(i => ({domain: i.domain as string, version: LongUtil.longToNumber(i.version!)}));

    this._graph = Graph.from(modelProto.graph!, graphInitializer);
  }

  private _graph: Graph;
  get graph(): Graph {
    return this._graph;
  }

  private _opsets: OpSet[];
  get opsets(): readonly OpSet[] {
    return this._opsets;
  }
}
