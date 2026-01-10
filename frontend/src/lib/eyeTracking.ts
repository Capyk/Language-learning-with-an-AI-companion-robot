/**
 * Browser-based eye-tracking using MediaPipe FaceLandmarker.
 * Implements calibration, gaze prediction, and smoothing logic.
 */

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { lusolve, matrix } from 'mathjs';

// MediaPipe landmark indices
const LEFT_IRIS_INDICES = [468, 469, 470, 471, 472];
const RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477];
const LEFT_EYE_INNER = 133;
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_INNER = 362;
const RIGHT_EYE_OUTER = 263;

export interface TrackingFrame {
    timestamp_ms: number;
    frame_idx: number;
    phase: string;
    screen_w_px: number;
    screen_h_px: number;
    face_detected: number;
    left_iris_x_norm: number | null;
    left_iris_y_norm: number | null;
    right_iris_x_norm: number | null;
    right_iris_y_norm: number | null;
    gaze_x_px: number | null;
    gaze_y_px: number | null;
    gaze_valid: number;
    on_screen: number;
}

export interface CalibrationPoint {
    x: number;
    y: number;
    features: number[];
}

class EyeTracker {
    private faceLandmarker: FaceLandmarker | null = null;
    private modelX: number[] | null = null;
    private modelY: number[] | null = null;
    private isCalibrated = false;

    // Smoothing state for on_screen
    private rawOnScreenHistory: { value: boolean; timestamp: number }[] = [];
    private currentOnScreen = 0;

    async initialize(): Promise<void> {
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                delegate: 'GPU'
            },
            numFaces: 1,
            runningMode: 'VIDEO',
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false
        });
    }

    private getIrisCenter(landmarks: NormalizedLandmark[], indices: number[]): [number, number] | null {
        try {
            const xCoords = indices.map(i => landmarks[i].x);
            const yCoords = indices.map(i => landmarks[i].y);
            const centerX = xCoords.reduce((a, b) => a + b, 0) / xCoords.length;
            const centerY = yCoords.reduce((a, b) => a + b, 0) / yCoords.length;
            return [centerX, centerY];
        } catch {
            return null;
        }
    }

    private buildFeatures(landmarks: NormalizedLandmark[], leftIris: [number, number], rightIris: [number, number]): number[] {
        const features = [leftIris[0], leftIris[1], rightIris[0], rightIris[1]];

        // Optional: relative features for robustness
        try {
            const leftInner = [landmarks[LEFT_EYE_INNER].x, landmarks[LEFT_EYE_INNER].y];
            const leftOuter = [landmarks[LEFT_EYE_OUTER].x, landmarks[LEFT_EYE_OUTER].y];
            const leftRelX = (leftIris[0] - leftInner[0]) / (leftOuter[0] - leftInner[0] + 1e-6);
            const leftRelY = leftIris[1] - (leftInner[1] + leftOuter[1]) / 2;

            const rightInner = [landmarks[RIGHT_EYE_INNER].x, landmarks[RIGHT_EYE_INNER].y];
            const rightOuter = [landmarks[RIGHT_EYE_OUTER].x, landmarks[RIGHT_EYE_OUTER].y];
            const rightRelX = (rightIris[0] - rightInner[0]) / (rightOuter[0] - rightInner[0] + 1e-6);
            const rightRelY = rightIris[1] - (rightInner[1] + rightOuter[1]) / 2;

            features.push(leftRelX, leftRelY, rightRelX, rightRelY);
        } catch {
            // If relative features fail, just use basic features
        }

        return features;
    }

    processFrame(videoElement: HTMLVideoElement, timestamp: number): {
        faceDetected: boolean;
        leftIrisX: number | null;
        leftIrisY: number | null;
        rightIrisX: number | null;
        rightIrisY: number | null;
        features: number[] | null;
    } {
        if (!this.faceLandmarker) {
            return { faceDetected: false, leftIrisX: null, leftIrisY: null, rightIrisX: null, rightIrisY: null, features: null };
        }

        const result = this.faceLandmarker.detectForVideo(videoElement, timestamp);

        if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
            return { faceDetected: false, leftIrisX: null, leftIrisY: null, rightIrisX: null, rightIrisY: null, features: null };
        }

        const landmarks = result.faceLandmarks[0];
        const leftIris = this.getIrisCenter(landmarks, LEFT_IRIS_INDICES);
        const rightIris = this.getIrisCenter(landmarks, RIGHT_IRIS_INDICES);

        if (!leftIris || !rightIris) {
            return { faceDetected: true, leftIrisX: null, leftIrisY: null, rightIrisX: null, rightIrisY: null, features: null };
        }

        const features = this.buildFeatures(landmarks, leftIris, rightIris);

        return {
            faceDetected: true,
            leftIrisX: leftIris[0],
            leftIrisY: leftIris[1],
            rightIrisX: rightIris[0],
            rightIrisY: rightIris[1],
            features
        };
    }

    calibrate(calibrationData: CalibrationPoint[]): void {
        if (calibrationData.length < 9) {
            throw new Error('Need at least 9 calibration points');
        }

        const X: number[][] = [];
        const yX: number[] = [];
        const yY: number[] = [];

        for (const point of calibrationData) {
            X.push([...point.features, 1]); // Add bias term
            yX.push(point.x);
            yY.push(point.y);
        }

        // Solve linear regression using mathjs
        const XMatrix = matrix(X);
        const yXMatrix = matrix(yX.map(v => [v]));
        const yYMatrix = matrix(yY.map(v => [v]));

        const modelXSolution = lusolve(XMatrix, yXMatrix);
        const modelYSolution = lusolve(XMatrix, yYMatrix);

        // Convert mathjs Matrix to number[]
        this.modelX = (modelXSolution as any).toArray().map((row: number[]) => row[0]);
        this.modelY = (modelYSolution as any).toArray().map((row: number[]) => row[0]);
        this.isCalibrated = true;
    }

    predictGaze(features: number[], screenW: number, screenH: number): {
        gazeX: number | null;
        gazeY: number | null;
        gazeValid: boolean;
    } {
        if (!this.isCalibrated || !this.modelX || !this.modelY) {
            return { gazeX: null, gazeY: null, gazeValid: false };
        }

        const featuresBias = [...features, 1];
        let gazeX = 0;
        let gazeY = 0;

        for (let i = 0; i < featuresBias.length; i++) {
            gazeX += featuresBias[i] * this.modelX[i];
            gazeY += featuresBias[i] * this.modelY[i];
        }

        const gazeValid = gazeX >= 0 && gazeX < screenW && gazeY >= 0 && gazeY < screenH;

        return { gazeX, gazeY, gazeValid };
    }

    computeOnScreen(faceDetected: boolean, gazeValid: boolean, featuresOk: boolean, currentTime: number): number {
        const rawOn = faceDetected && gazeValid && featuresOk;

        // Add to history
        this.rawOnScreenHistory.push({ value: rawOn, timestamp: currentTime });

        // Keep only last 2 seconds of history
        const cutoff = currentTime - 2000;
        this.rawOnScreenHistory = this.rawOnScreenHistory.filter(h => h.timestamp >= cutoff);

        // Hysteresis logic
        if (this.currentOnScreen === 1) {
            // Currently ON -> check if we should turn OFF
            // Need raw_on == false for >= 700ms
            const recentHistory = this.rawOnScreenHistory.filter(h => h.timestamp >= currentTime - 700);
            const allFalse = recentHistory.every(h => !h.value);
            if (allFalse && recentHistory.length > 0) {
                this.currentOnScreen = 0;
            }
        } else {
            // Currently OFF -> check if we should turn ON
            // Need raw_on == true for >= 250ms
            const recentHistory = this.rawOnScreenHistory.filter(h => h.timestamp >= currentTime - 250);
            const allTrue = recentHistory.every(h => h.value);
            if (allTrue && recentHistory.length > 0) {
                this.currentOnScreen = 1;
            }
        }

        return this.currentOnScreen;
    }

    close(): void {
        if (this.faceLandmarker) {
            this.faceLandmarker.close();
            this.faceLandmarker = null;
        }
    }
}

export default EyeTracker;
