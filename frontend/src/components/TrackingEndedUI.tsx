import React from 'react';

interface TrackingEndedUIProps {
    sessionId: string;
    onContinue: () => void;
}

const TrackingEndedUI: React.FC<TrackingEndedUIProps> = ({ sessionId, onContinue }) => {
    const handleDownload = () => {
        // Trigger download from backend
        window.location.href = `/experiment/eye_tracking/export/${sessionId}.xlsx`;
    };

    return (
        <div className="w-full max-w-2xl bg-white rounded-[2.5rem] shadow-2xl p-12 text-center border-8 border-white">
            {/* Icon */}
            <div className="mb-8">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="80"
                    height="80"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-blue-500 mx-auto"
                >
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                </svg>
            </div>

            {/* Message */}
            <h1 className="text-4xl font-black text-slate-800 mb-4">Tracking has ended</h1>
            <p className="text-lg text-slate-600 mb-8">
                Eye-tracking data has been collected and is ready for download.
            </p>

            {/* Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button
                    onClick={handleDownload}
                    className="px-8 py-4 bg-green-600 text-white rounded-2xl font-bold text-lg shadow-lg hover:bg-green-700 active:scale-95 transition-all flex items-center justify-center gap-2"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="20"
                        height="20"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Download Excel
                </button>

                <button
                    onClick={onContinue}
                    className="px-8 py-4 bg-slate-800 text-white rounded-2xl font-bold text-lg shadow-lg hover:bg-slate-700 active:scale-95 transition-all"
                >
                    Continue the test
                </button>
            </div>
        </div>
    );
};

export default TrackingEndedUI;
