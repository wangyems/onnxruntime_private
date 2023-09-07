// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {resolveBackend} from './backend-impl.js';
import {TrainingSessionHandler} from './backend.js';
import {InferenceSession as InferenceSession} from './inference-session.js';
import {processModel, processModelOrOptions} from './session-impl-utils.js';
import {TrainingSession as TrainingSessionInterface} from './training-session.js';

type SessionOptions = InferenceSession.SessionOptions;

export class TrainingSession implements TrainingSessionInterface {
  private constructor(handler: TrainingSessionHandler) {
    this.handler = handler;
  }

  async release(): Promise<void> {
    return this.handler.dispose();
  }

  static create(checkpointStateUri: string, trainModelURI: string, options?: InferenceSession.SessionOptions):
      Promise<TrainingSession>;
  static create(
      checkpointState: string|ArrayBufferLike|Uint8Array, trainModelData: string|ArrayBufferLike|Uint8Array,
      evalModelData?: string|ArrayBufferLike|Uint8Array, optimizerModelData?: string|ArrayBufferLike|Uint8Array,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;
  static async create(
      arg0: string|ArrayBufferLike|Uint8Array, arg1: string|ArrayBufferLike|Uint8Array,
      arg2?: string|ArrayBufferLike|Uint8Array|InferenceSession.SessionOptions,
      arg3?: string|ArrayBufferLike|Uint8Array, arg4?: InferenceSession.SessionOptions): Promise<TrainingSession> {
    // optional fields:
    let options: SessionOptions = {};
    const checkpointState: string|Uint8Array = processModel(arg0);
    const trainModel: string|Uint8Array = processModel(arg1);
    let evalModel: string|Uint8Array = '';
    let optimizerModel: string|Uint8Array = '';

    if (typeof arg2 !== 'undefined') {
      const [return1, return2] = processModelOrOptions(arg2, options, evalModel);
      options = return1;
      evalModel = return2;
    }
    if (typeof arg3 !== 'undefined') {
      optimizerModel = processModel(arg3);
    }
    if (typeof arg4 !== 'undefined') {
      options = arg4;
    }

    // get backend hints
    const eps = options.executionProviders || [];
    const backendHints = eps.map(i => typeof i === 'string' ? i : i.name);
    const backend = await resolveBackend(backendHints);
    if (backend.createTrainingSessionHandler) {
      const handler =
          await backend.createTrainingSessionHandler(checkpointState, trainModel, evalModel, optimizerModel, options);
      return new TrainingSession(handler);
    } else {
      throw new Error(
          'Training backend could not be resolved. ' +
          'Make sure you\'re using the correct configuration + webassembly files.');
    }
  }

  get inputNames(): readonly string[] {
    return this.handler.inputNames;
  }
  get outputNames(): readonly string[] {
    return this.handler.outputNames;
  }
  private handler: TrainingSessionHandler;
}
