// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Backend, InferenceSession, SessionHandler, TrainingSessionHandler} from 'onnxruntime-common';

import {Binding, binding} from './binding';

class OnnxruntimeSessionHandler implements SessionHandler {
  #inferenceSession: Binding.InferenceSession;

  constructor(pathOrBuffer: string|Uint8Array, options: InferenceSession.SessionOptions) {
    this.#inferenceSession = new binding.InferenceSession();
    if (typeof pathOrBuffer === 'string') {
      this.#inferenceSession.loadModel(pathOrBuffer, options);
    } else {
      this.#inferenceSession.loadModel(pathOrBuffer.buffer, pathOrBuffer.byteOffset, pathOrBuffer.byteLength, options);
    }
    this.inputNames = this.#inferenceSession.inputNames;
    this.outputNames = this.#inferenceSession.outputNames;
  }

  async dispose(): Promise<void> {
    return Promise.resolve();
  }

  readonly inputNames: string[];
  readonly outputNames: string[];

  startProfiling(): void {
    // TODO: implement profiling
  }
  endProfiling(): void {
    // TODO: implement profiling
  }

  async run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType, options: InferenceSession.RunOptions):
      Promise<SessionHandler.ReturnType> {
    return new Promise((resolve, reject) => {
      process.nextTick(() => {
        try {
          resolve(this.#inferenceSession.run(feeds, fetches, options));
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }
}

class OnnxruntimeBackend implements Backend {
  async init(): Promise<void> {
    return Promise.resolve();
  }

  async createSessionHandler(pathOrBuffer: string|Uint8Array, options?: InferenceSession.SessionOptions):
      Promise<SessionHandler> {
    return new Promise((resolve, reject) => {
      process.nextTick(() => {
        try {
          resolve(new OnnxruntimeSessionHandler(pathOrBuffer, options || {}));
        } catch (e) {
          // reject if any error is thrown
          reject(e);
        }
      });
    });
  }

  /* eslint-disable @typescript-eslint/no-unused-vars */
  async createTrainingSessionHandler(
      checkpointStateUriOrBuffer: string|Uint8Array, trainModelUriOrBuffer: string|Uint8Array,
      evalModelUriOrBuffer: string|Uint8Array, optimizerModelUriOrBuffer: string|Uint8Array,
      options: InferenceSession.SessionOptions): Promise<TrainingSessionHandler> {
    throw new Error('Training not supported on Nodejs');
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */
}

export const onnxruntimeBackend = new OnnxruntimeBackend();
export const listSupportedBackends = binding.listSupportedBackends;
