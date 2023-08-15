// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session.js';
import {OnnxValue} from './onnx-value.js';

/**
 * @internal
 */
export declare namespace SessionHandler {
  type FeedsType = {[name: string]: OnnxValue};
  type FetchesType = {[name: string]: OnnxValue | null};
  type ReturnType = {[name: string]: OnnxValue};
}

/**
 * Represent a handler instance of an inference session.
 *
 * @internal
 */
export interface SessionHandler {
  dispose(): Promise<void>;

  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];

  startProfiling(): void;
  endProfiling(): void;

  run(feeds: SessionHandler.FeedsType, fetches: SessionHandler.FetchesType,
      options: InferenceSession.RunOptions): Promise<SessionHandler.ReturnType>;
}

/**
 * Represent a backend that provides implementation of model inferencing.
 *
 * @internal
 */
export interface Backend {
  /**
   * Initialize the backend asynchronously. Should throw when failed.
   */
  init(): Promise<void>;

  createSessionHandler(uriOrBufferOrResponses: string|[string,string]|Uint8Array|[Uint8Array, ArrayBuffer],
    options?: InferenceSession.SessionOptions):
      Promise<SessionHandler>;
}

export {registerBackend} from './backend-impl.js';
