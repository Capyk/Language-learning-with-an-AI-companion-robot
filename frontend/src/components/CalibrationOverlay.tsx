import React, { useState, useEffect, useRef } from 'react';
import EyeTracker from '../lib/eyeTracking';
import type { CalibrationPoint } from '../lib/eyeTracking';

interface CalibrationOverlayProps {
    onCalibrationComplete: (tracker: EyeTracker) => void;
    onAbort: () => void;
}

const CalibrationOverlay: React.FC<CalibrationOverlayProps> = ({ onCalibrationComplete, onAbort }) => {
    const [currentPointIndex, setCurrentPointIndex] = useState(0);
    const [waitingForSpace, setWaitingForSpace] = useState(true);
    const [faceDetected, setFaceDetected] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const videoRef = useRef<HTMLVideoElement>(null);
    const trackerRef = useRef<EyeTracker | null>(null);
    const calibrationDataRef = useRef<CalibrationPoint[]>([]);
    const collectedFramesRef = useRef<number[][]>([]);
    const animationFrameRef = useRef<number | null>(null);
    const timeoutRef = useRef<number | null>(null);

    // Generate 9 calibration points (3x3 grid, 10% margin)
    const calibrationPoints = useRef<{ x: number; y: number }[]>([]);

    useEffect(() => {
        const screenW = Math.round(window.screen.width * window.devicePixelRatio);
        const screenH = Math.round(window.screen.height * window.devicePixelRatio);
        const marginX = 0.1;
        const marginY = 0.1;

        const xPositions = [
            Math.round(screenW * marginX),
            Math.round(screenW * 0.5),
            Math.round(screenW * (1 - marginX))
        ];
        const yPositions = [
            Math.round(screenH * marginY),
            Math.round(screenH * 0.5),
            Math.round(screenH * (1 - marginY))
        ];

        const points: { x: number; y: number }[] = [];
        for (const y of yPositions) {
            for (const x of xPositions) {
                points.push({ x, y });
            }
        }
        calibrationPoints.current = points;
    }, []);

    useEffect(() => {
        const initializeTracking = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play();
                }

                const tracker = new EyeTracker();
                await tracker.initialize();
                trackerRef.current = tracker;

                startProcessingLoop();
            } catch (err) {
                setError('Camera access denied or MediaPipe failed to load');
                console.error(err);
            }
        };

        initializeTracking();

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
            }
            if (trackerRef.current) {
                trackerRef.current.close();
            }
        };
    }, []);

    const startProcessingLoop = () => {
        const processFrame = () => {
            if (!videoRef.current || !trackerRef.current) return;

            const timestamp = performance.now();
            const result = trackerRef.current.processFrame(videoRef.current, timestamp);
            setFaceDetected(result.faceDetected);

            if (!waitingForSpace && result.faceDetected && result.features) {
                collectedFramesRef.current.push(result.features);
                setProgress(collectedFramesRef.current.length);

                if (collectedFramesRef.current.length >= 15) {
                    // Average features and save calibration point
                    const avgFeatures = collectedFramesRef.current[0].map((_, i) =>
                        collectedFramesRef.current.reduce((sum, f) => sum + f[i], 0) / collectedFramesRef.current.length
                    );

                    const point = calibrationPoints.current[currentPointIndex];
                    calibrationDataRef.current.push({ x: point.x, y: point.y, features: avgFeatures });

                    // Move to next point
                    if (currentPointIndex < 8) {
                        setCurrentPointIndex(currentPointIndex + 1);
                        setWaitingForSpace(true);
                        setProgress(0);
                        collectedFramesRef.current = [];
                    } else {
                        // Calibration complete
                        try {
                            trackerRef.current.calibrate(calibrationDataRef.current);
                            onCalibrationComplete(trackerRef.current);
                        } catch (err) {
                            setError('Calibration failed');
                            console.error(err);
                        }
                        return;
                    }
                }
            }

            animationFrameRef.current = requestAnimationFrame(processFrame);
        };

        processFrame();
    };

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onAbort();
            } else if (e.key === ' ' && waitingForSpace && faceDetected) {
                e.preventDefault();
                setWaitingForSpace(false);
                collectedFramesRef.current = [];
                setProgress(0);

                // Start timeout (5 seconds)
                timeoutRef.current = setTimeout(() => {
                    if (collectedFramesRef.current.length < 15) {
                        setError('Timeout - retrying point');
                        setTimeout(() => {
                            setError(null);
                            setWaitingForSpace(true);
                            collectedFramesRef.current = [];
                            setProgress(0);
                        }, 2000);
                    }
                }, 5000);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [waitingForSpace, faceDetected, onAbort]);

    const currentPoint = calibrationPoints.current[currentPointIndex];

    return (
        <div className="fixed inset-0 bg-black z-[9999] flex items-center justify-center">
            {/* Hidden video element */}
            <video ref={videoRef} className="hidden" />

            {/* Calibration point */}
            {currentPoint && (
                <div
                    className="absolute w-10 h-10 rounded-full bg-white border-4 border-gray-300"
                    style={{
                        left: `${(currentPoint.x / (window.screen.width * window.devicePixelRatio)) * 100}%`,
                        top: `${(currentPoint.y / (window.screen.height * window.devicePixelRatio)) * 100}%`,
                        transform: 'translate(-50%, -50%)'
                    }}
                />
            )}

            {/* Status overlay */}
            <div className="absolute top-12 left-12 text-white space-y-4">
                <p className="text-2xl font-bold">Punkt {currentPointIndex + 1}/9</p>
                <p className={`text-xl font-bold ${faceDetected ? 'text-green-400' : 'text-red-400'}`}>
                    Face detected: {faceDetected ? 'YES' : 'NO'}
                </p>
                {waitingForSpace ? (
                    <p className="text-lg">Schau auf den Punkt und drücke SPACE</p>
                ) : (
                    <p className="text-lg text-yellow-300">Sammle Daten... {progress}/15</p>
                )}
                {error && <p className="text-red-500 text-lg font-bold">{error}</p>}
                <p className="text-sm text-gray-400">ESC = Abbrechen</p>
            </div>
        </div>
    );
};

export default CalibrationOverlay;
