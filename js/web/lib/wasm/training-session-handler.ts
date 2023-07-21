import {readFile} from 'fs';
import {TrainingHandler} from 'onnxruntime-common';
import { promisify } from 'util';
import * as core from './wasm-core-impl';

export class OnnxruntimeWebAssemblyTrainingHandler implements TrainingHandler {
    handlerId: number;
//     private sessionId: number;
//     private checkpoint: CheckpointState;

//     inputNames: string[];
//     outputNames: string[];

    disposeCheckpointState(): Promise<void> {
        core.releaseCheckpoint(this.handlerId);
        throw new Error('after releaseCheckpoint call');
    }

    disposeTrainingSession(): Promise<void> {
        throw new Error('Method not implemented.');
    }

    disposeHandler(): Promise<void> {
        throw new Error('Method not implemented.');
    }

    async loadCheckpointAllocate(path: string): Promise<number> {
        const response = await fetch(path);
        const arrayBuffer = await response.arrayBuffer();
        return this.loadCheckpoint(new Uint8Array(arrayBuffer));
    }

    async loadCheckpoint(pathOrBuffer: string|Uint8Array): Promise<number> {
        console.log("inside lib wasm handler load checkpoint");
        if (typeof pathOrBuffer ==='string') {
            if (typeof fetch === 'undefined') {
                // node
                const checkpointData = await promisify(readFile)(pathOrBuffer);
                this.handlerId = await core.loadCheckpoint(checkpointData);
                return this.handlerId;
            } else {
                return this.loadCheckpointAllocate(pathOrBuffer);
            }
        } else {
            this.handlerId = await core.loadCheckpoint(pathOrBuffer);
            return this.handlerId;
        }
    }

}
