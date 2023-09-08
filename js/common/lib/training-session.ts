// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceSession} from './inference-session.js';
import {TrainingSession as TrainingSessionImpl} from './training-session-impl.js';

/* eslint-disable @typescript-eslint/no-redeclare */

/**
 * Represent a runtime instance of an ONNX training session,
 * which contains a model that can be trained, and, optionally,
 * an eval and optimizer model.
 */
export interface TrainingSession {
  // #region release()

  /**
   * Release the inference session and the underlying resources.
   */
  release(): Promise<void>;
  // #endregion

  // #region metadata

  /**
   * Get input names of the loaded model.
   */
  readonly inputNames: readonly string[];

  /**
   * Get output names of the loaded model.
   */
  readonly outputNames: readonly string[];
  // #endregion
}

export interface TrainingSessionFactory {
  // #region create()

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param uri - The URI or file path of the model to load.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(checkpointStateUri: string, trainModelURI: string, options?: InferenceSession.SessionOptions):
      Promise<TrainingSession>;

  /**
   * Create a new training session and load models asynchronously from an ONNX model file.
   *
   * @param uri - The URI or file path of the model to load.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateUri: string, trainModelURI: string, evalModelURI?: string, optimizerModelURI?: string,
      options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load model asynchronously from an array bufer.
   *
   * @param buffer - An ArrayBuffer representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointStateBuffer: ArrayBufferLike, trainModelBuffer: ArrayBufferLike, evalModelBuffer?: ArrayBufferLike,
      optimizerModelBuffer?: ArrayBufferLike, options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  /**
   * Create a new training session and load model asynchronously from a Uint8Array.
   *
   * @param buffer - A Uint8Array representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
      checkpointState: Uint8Array, trainModelData: Uint8Array, evalModelData?: Uint8Array,
      optimizerModelData?: Uint8Array, options?: InferenceSession.SessionOptions): Promise<TrainingSession>;

  // #endregion
}

// eslint-disable-next-line @typescript-eslint/naming-convention
export const TrainingSession: TrainingSessionFactory = TrainingSessionImpl;
