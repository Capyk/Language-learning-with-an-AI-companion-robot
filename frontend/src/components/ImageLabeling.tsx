import React, { useState, useEffect } from 'react';

// Renamed to ImageLabeling (matching user's import name)
const ImageLabeling = () => {
    const [index, setIndex] = useState(0); // Current image index (0-indexed)
    const [imageData, setImageData] = useState<{ image_url: string; topic: string; german_label: string; english_label: string } | null>(null);
    const [userInput, setUserInput] = useState('');
    const [feedback, setFeedback] = useState<{ correction: string; tip: string; correct_label: string } | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [quizMode, setQuizMode] = useState(false); // NEW STATE: Controls the mode

    // Define the exact width of the content block
    const contentWidth = 400; 
    const contentPadding = 20; 
    const totalImages = 5; 

    // --- Core Function: Fetch the current image data ---
    const fetchImage = async () => {
        if (index >= totalImages) {
            setImageData(null);
            return;
        }

        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch(`/image-labeling/${index}`);
            
            if (response.status === 404) {
                setImageData(null);
                return;
            }
            if (!response.ok) throw new Error('Failed to fetch image data');
            
            const data = await response.json();
            
            setImageData({
                image_url: data.image_url, 
                topic: data.topic,
                german_label: data.german_label || data.topic, 
                english_label: data.english_label || data.topic,
            });
            setFeedback(null); 
            setUserInput('');
        } catch (err) {
            console.error(err);
            setError("Connection error or image index out of bounds.");
            setImageData(null); 
        } finally {
            setIsLoading(false);
        }
    };

    // Handle "Next" button click
    const handleNext = () => {
        if (!isLoading && index < totalImages) {
            // In quiz mode, you might only allow 'Next' if the question was answered correctly or attempted.
            // For now, we allow moving forward.
            setIndex((prevIndex) => prevIndex + 1);
        }
    };

    // Handle "Previous" button click
    const handlePrevious = () => {
        if (!isLoading && index > 0) {
            setIndex((prevIndex) => prevIndex - 1);
        }
    };

    // NEW FUNCTION: Restart Viewing Mode
    const handleRestart = () => {
        setIndex(0);
        setQuizMode(false);
    };

    // NEW FUNCTION: Start Quiz Mode
    const handleStartQuiz = () => {
        setIndex(0); // Reset to first image for the quiz
        setQuizMode(true);
    };
    
    // Fetch image data when index changes or on component mount
    useEffect(() => {
        fetchImage();
    }, [index]);

    // --- Core Function: Analyze the user's input (Now uses API) ---
    const analyzeInput = async () => {
        if (!userInput.trim() || !imageData) return;
        
        setIsLoading(true);
        setError(null);

        try {
            // *** API CALL TO BACKEND FOR GEMINI FEEDBACK ***
            const response = await fetch(`/image-labeling/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    // Send user input and the known correct label for Gemini context
                    user_input: userInput,
                    user_label: imageData.german_label, 
                    user_id: "demo_user_1", // Placeholder
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch correction from API.');
            }

            const data = await response.json();
            
            // Set feedback state using data returned from Gemini via FastAPI
            setFeedback({
                correction: data.correction, // e.g., "Incorrect Article" or "Correct!"
                tip: data.tip,               // Detailed tip from Gemini
                correct_label: data.correct_label // Ground truth label
            });

        } catch (err) {
            console.error("API Analyze Error:", err);
            setError(`Correction service error: ${err.message}`);
        } finally {
            setIsLoading(false);
        }
    };


    // --- RENDERING LOGIC ---
    return (
        <div style={{ 
            minHeight: '100vh', 
            width: '100vw', 
            padding: '20px',
            backgroundColor: '#1E1E1E', 
            color: '#FFFFFF',
            position: 'relative', 
            overflowX: 'hidden', 
        }}>
            
            {/* 2. INNER CONTENT BLOCK: ABSOLUTE CENTERING FIX */}
            <div style={{ 
                textAlign: 'left',
                width: `${contentWidth}px`, 
                padding: `0 ${contentPadding}px`, 
                position: 'absolute', 
                left: '50%', 
                transform: 'translateX(-50%)', 
                top: '20px', 
            }}>
                {/* Main Heading */}
                <h1 style={{fontSize: '2.5em', marginBottom: '40px', color: '#FFFFFF'}}>Language Learning with AI</h1>
                {/* Sub-Heading */}
                <h2 style={{fontSize: '1.8em', marginBottom: '10px'}}>Image Labeling Task</h2>
                {isLoading && <p>Loading...</p>}
                {error && <p style={{color: '#ff4d4d'}}>Error: {error}</p>}
                
                {/* --- IMAGE AND LABEL CONTENT --- */}
                {imageData ? (
                    <>
                        {/* Labels (Content changes based on Quiz Mode) */}
                        <p style={{marginTop: '15px', fontSize: '1.1em'}}><strong>Topic:</strong> {imageData.topic}</p>
                        <p style={{fontSize: '1.1em'}}>
                            {/* In Quiz Mode, hide the German label */}
                            <strong>German Label:</strong> {quizMode ? '???' : imageData.german_label}
                        </p>
                        <p style={{fontSize: '1.1em'}}><strong>English Label:</strong> {imageData.english_label}</p>
                        
                        <img 
                            src={imageData.image_url} 
                            alt="Labeling Task" 
                            onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = `https://placehold.co/300x180/4F46E5/FFFFFF?text=${imageData.topic}`; }}
                            style={{ 
                                maxWidth: '250px', 
                                height: 'auto',
                                marginTop: '20px',
                                marginBottom: '10px',
                                border: '1px solid #444' 
                            }} 
                        />
                        
                        {/* Quiz Input Field (Only visible in Quiz Mode) */}
                        {quizMode && (
                            <div style={{marginTop: '20px', marginBottom: '20px'}}>
                                <input
                                    type="text"
                                    placeholder="Enter German label (e.g., die Wurst)"
                                    value={userInput}
                                    onChange={(e) => setUserInput(e.target.value)}
                                    // Disable after receiving a "Correct!" response
                                    disabled={isLoading || feedback?.correction?.includes("Correct")} 
                                    style={{ 
                                        padding: '10px', 
                                        width: '250px', 
                                        border: '1px solid #555',
                                        backgroundColor: '#333',
                                        color: 'white'
                                    }}
                                />
                                <button 
                                    onClick={analyzeInput} 
                                    // Disable if loading, input is empty, or already correct
                                    disabled={isLoading || !userInput.trim() || feedback?.correction?.includes("Correct")} 
                                    style={{ 
                                        marginLeft: '10px', 
                                        padding: '10px 20px', 
                                        backgroundColor: '#4CAF50', 
                                        color: 'white', 
                                        border: 'none', 
                                        borderRadius: '4px',
                                        cursor: 'pointer'
                                    }}
                                >
                                    Submit
                                </button>
                                {feedback && (
                                    <div style={{ marginTop: '10px', padding: '10px', border: '1px solid #444', backgroundColor: '#333' }}>
                                        {/* Use the full correction message from Gemini */}
                                        <p style={{marginBottom: '5px'}}><strong>{feedback.correction.includes("Correct") ? '✅' : '❌'} Correction:</strong> {feedback.correction}</p>
                                        
                                        {/* Display Gemini's detailed tip */}
                                        <p style={{color: '#88aaff'}}><strong>💡 Tip:</strong> {feedback.tip}</p>

                                        {/* Only show the correct label if the answer was incorrect */}
                                        {!feedback.correction.includes("Correct") && <p style={{marginTop: '5px'}}><strong>Correct Answer:</strong> {feedback.correct_label}</p>}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Placeholder for fixed area */}
                        <div style={{ height: '70px', marginTop: '20px' }} /> 
                    </>
                ) : (
                    // --- QUIZ FINISHED / VIEWING FINISHED SCREEN ---
                    <div style={{marginTop: '50px', textAlign: 'center'}}>
                        <p style={{fontSize: '1.2em', marginBottom: '30px'}}>
                            {quizMode ? 'Quiz Finished! Your scores would go here.' : 'Viewing Mode Finished.'}
                        </p>
                        <button onClick={handleRestart} style={{ 
                            padding: '10px 20px', 
                            backgroundColor: '#ffc107', 
                            color: '#333', 
                            border: 'none', 
                            borderRadius: '4px', 
                            cursor: 'pointer',
                            marginRight: '10px'
                        }}>
                            🔄 Restart Viewing
                        </button>
                        {!quizMode && (
                            <button onClick={handleStartQuiz} style={{ 
                                padding: '10px 20px', 
                                backgroundColor: '#17a2b8', 
                                color: 'white', 
                                border: 'none', 
                                borderRadius: '4px', 
                                cursor: 'pointer',
                            }}>
                                🧠 Start Quiz Mode
                            </button>
                        )}
                    </div>
                )}
            </div>

            {/* 4. FIXED NAVIGATION BUTTON WRAPPER */}
            {index < totalImages && imageData && (
                <div style={{
                    position: 'fixed',
                    bottom: '20px', 
                    left: '50%', 
                    transform: 'translateX(-50%)', 
                    width: `${contentWidth}px`, 
                    zIndex: 1000,
                    
                    display: 'flex',
                    justifyContent: 'center',
                    padding: `0 ${contentPadding}px`, 
                }}>
                    
                    {/* Previous Button (Only visible if index > 0) */}
                    {index > 0 && (
                        <button onClick={handlePrevious} disabled={isLoading} style={{ 
                            padding: '10px 20px', 
                            backgroundColor: '#6c757d', 
                            color: 'white', 
                            border: 'none', 
                            borderRadius: '4px', 
                            cursor: 'pointer',
                            marginRight: '10px', 
                        }}>
                            Previous Image ({index} / {totalImages})
                        </button>
                    )}

                    {/* Next Button (Becomes "Next Question" in quiz mode) */}
                    <button onClick={handleNext} disabled={isLoading} style={{ 
                        padding: '10px 20px', 
                        backgroundColor: '#007bff', 
                        color: 'white', 
                        border: 'none', 
                        borderRadius: '4px', 
                        cursor: 'pointer',
                    }}>
                        {quizMode ? 'Next Question' : `Next Image (${index + 1} / ${totalImages})`}
                    </button>
                </div>
            )}
        </div>
    );
};


export default ImageLabeling;