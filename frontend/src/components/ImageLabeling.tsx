import React, { useState, useEffect, useCallback } from 'react';

// --- Icons (Inline SVG Components to avoid lucide-react dependency) ---
const Icon = {
  Send: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
  ),
  CheckCircle: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
  ),
  XCircle: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
  ),
  ArrowRight: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
  ),
  BookOpen: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
  ),
  GraduationCap: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
  ),
  PlayCircle: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
  ),
  Info: ({ size = 24, className = "" }: { size?: number, className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
  )
};

// --- Types & Interfaces ---

interface TrialData {
  phase: 'pre-test' | 'learning' | 'post-test';
  index: number;
  total_in_phase: number;
  task_type: 'article_mcq' | 'plural_mcq' | 'type_word';
  image_url: string;
  english_gloss: string;
  options?: string[];
  german_word?: string;
  status?: string;
}

interface FeedbackData {
  is_correct: boolean;
  feedback: string;
  example?: string;
  reveal?: boolean;
  move_next: boolean;
}

interface SessionData {
  session_id: string;
  condition: 'A' | 'B';
}

const ImageLabeling: React.FC = () => {
  // --- State Management ---
  const [session, setSession] = useState<{ session_id: string; condition: string } | null>(null);
  const [currentTrial, setCurrentTrial] = useState<TrialData | null>(null);
  const [userInput, setUserInput] = useState('');
  const [selectedArticle, setSelectedArticle] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<FeedbackData | null>(null);
  const [startTime, setStartTime] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  
  // --- UI/UX State ---
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTransition, setShowTransition] = useState(false);
  const [nextPhaseName, setNextPhaseName] = useState('');
  const [localAttempt, setLocalAttempt] = useState(1);

  // Global background color reset to ensure proper centering and color match with the theme
  useEffect(() => {
    document.body.style.backgroundColor = '#f8fafc'; // bg-slate-50
    document.body.style.margin = '0';
    document.body.style.display = 'flex';
    document.body.style.alignItems = 'center';
    document.body.style.justifyContent = 'center';
    document.body.style.minHeight = '100vh';
    return () => { document.body.style.backgroundColor = ''; };
  }, []);

  // 1. Initialize Experiment Session
  const startExperiment = async (condition: 'A' | 'B') => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/experiment/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: `user_${Date.now()}`, condition }),
      });
      if (!response.ok) throw new Error("Initialization failed");
      const data = await response.json();
      setSession(data);
      // Immediately fetch the first trial for the new session
      await fetchNextTrial(data.session_id);
    } catch (err) {
      setError("Unable to connect to backend server. Please check if it is running.");
    } finally {
      setIsLoading(false);
    }
  };

  // 2. Fetch Next Trial (Handles phase transitions)
  const fetchNextTrial = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setFeedback(null);
    setUserInput('');
    setSelectedArticle(null);
    setLocalAttempt(1);
    try {
      const response = await fetch(`/experiment/trial/${sessionId}`);
      const data: TrialData = await response.json();
      
      if (data.status === "completed") {
        setIsComplete(true);
      } else {
        // Detect phase change to show the transition screen
        if (currentTrial && currentTrial.phase !== data.phase) {
          setNextPhaseName(data.phase);
          setShowTransition(true);
          setCurrentTrial(data);
        } else {
          setCurrentTrial(data);
          setStartTime(Date.now() / 1000);
        }
      }
    } catch (err) {
      setError("Error fetching the next task.");
    } finally {
      setIsLoading(false);
    }
  }, [currentTrial]);

  // 3. Submit User Answer
  const submitAnswer = async (answer?: string) => {
    let finalAnswer = "";
    
    // Typing Task Logic: Combines selected article button + text input
    if (currentTrial?.task_type === 'type_word') {
      if (!selectedArticle) {
          setError("Please select an article (der/die/das) first!");
          return;
      }
      if (!userInput.trim()) return;

      const trimmedInput = userInput.trim();

      // Noun Capitalization Check (Grammar rule requirement)
      // UPDATED: This rule only applies during the LEARNING phase for Condition B (Adaptive)
      if (currentTrial?.phase === 'learning' && session?.condition === 'B') {
        if (trimmedInput[0] !== trimmedInput[0].toUpperCase()) {
            setFeedback({
                is_correct: false,
                feedback: "Grammar Note: All German nouns must start with a Capital Letter! Please correct your input.",
                move_next: false
            });
            return;
        }
      }

      finalAnswer = `${selectedArticle} ${trimmedInput}`;
    } else {
      // MCQ Logic: Uses the clicked button value
      finalAnswer = answer || userInput;
    }

    if (!finalAnswer) return;

    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/experiment/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session?.session_id,
          user_answer: finalAnswer,
          start_time: startTime
        }),
      });
      const data: FeedbackData = await response.json();
      setFeedback(data);

      if (data.move_next) {
        // Delay moving to the next item so the user can read the feedback
        setTimeout(() => { if (session) fetchNextTrial(session.session_id); }, 2500);
      } else {
        setLocalAttempt(prev => prev + 1);
      }
    } catch (err) {
      setError("Error submitting your answer.");
    } finally {
      setIsLoading(false);
    }
  };

  // --- UI Helpers ---

  const insertChar = (char: string) => setUserInput(prev => prev + char);

  const GermanCharPicker = () => (
    <div className="flex gap-2 mt-3">
      {['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'].map(char => (
        <button key={char} onClick={() => insertChar(char)} className="w-10 h-10 flex items-center justify-center bg-slate-50 border border-slate-200 rounded-lg hover:bg-white hover:border-blue-400 font-bold text-slate-500 transition-all shadow-sm">{char}</button>
      ))}
    </div>
  );

  // --- Main Render States ---

  if (isComplete) {
    return (
      <div className="w-[600px] min-h-[500px] flex flex-col items-center justify-center bg-white rounded-[3rem] shadow-2xl text-center p-12 border-8 border-white">
        <div className="bg-green-50 p-6 rounded-full mb-6"><Icon.CheckCircle size={64} className="text-green-500" /></div>
        <h1 className="text-4xl font-black text-slate-700 mb-4 tracking-tight">Well Done!</h1>
        <p className="text-slate-500 text-lg mb-8 leading-relaxed">Experiment complete. Thank you for participating!</p>
        <button onClick={() => window.location.reload()} className="px-10 py-4 bg-blue-600 text-white rounded-2xl font-bold hover:bg-blue-700 transition-all shadow-lg active:scale-95">Restart</button>
      </div>
    );
  }

  if (showTransition) {
    return (
      <div className="w-[600px] min-h-[500px] flex flex-col items-center justify-center bg-white rounded-[3rem] shadow-2xl text-center p-12 border-8 border-white">
        <div className="bg-blue-50 p-6 rounded-full mb-6">
          {nextPhaseName === 'learning' ? <Icon.BookOpen size={64} className="text-blue-500" /> : <Icon.GraduationCap size={64} className="text-purple-400" />}
        </div>
        <h2 className="text-3xl font-black text-slate-700 mb-4 capitalize">Starting {nextPhaseName}</h2>
        <p className="text-slate-500 text-lg mb-10 leading-relaxed">
            {nextPhaseName === 'learning' 
                ? "Let's start practicing! You'll get hints and tips as you go along." 
                : "Great job learning. Now let's see how much you remember in the final test."}
        </p>
        <button onClick={() => { setShowTransition(false); setStartTime(Date.now() / 1000); }} className="flex items-center gap-2 px-10 py-4 bg-slate-800 text-white rounded-2xl font-bold hover:bg-slate-700 transition-all active:scale-95 shadow-xl">Start Phase <Icon.PlayCircle size={20} /></button>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="w-[600px] text-center flex flex-col items-center">
        <h1 className="text-4xl font-black mb-4 text-slate-800 tracking-tight">German Vocab Study</h1>
        <p className="text-slate-500 mb-12 text-lg">Choose a condition to begin the session.</p>
        <div className="grid grid-cols-1 gap-6 w-full max-w-md">
          <button onClick={() => startExperiment('A')} className="p-8 bg-white border-4 border-transparent hover:border-blue-400 rounded-[2.5rem] shadow-xl transition-all group text-left flex items-start gap-6">
            <div className="bg-blue-50 p-4 rounded-2xl text-blue-400 group-hover:bg-blue-500 group-hover:text-white transition-colors"><Icon.XCircle size={32} /></div>
            <div>
              <h2 className="text-2xl font-bold text-slate-700 mb-1">Group A</h2>
              <p className="text-slate-500">Static feedback (Instant corrections only).</p>
            </div>
          </button>
          <button onClick={() => startExperiment('B')} className="p-8 bg-white border-4 border-transparent hover:border-purple-400 rounded-[2.5rem] shadow-xl transition-all group text-left flex items-start gap-6">
            <div className="bg-purple-50 p-4 rounded-2xl text-purple-600 group-hover:bg-purple-600 group-hover:text-white transition-colors"><Icon.CheckCircle size={32} /></div>
            <div>
              <h2 className="text-2xl font-bold text-slate-700 mb-1">Group B</h2>
              <p className="text-slate-500">Adaptive feedback (Scaffolding hints).</p>
            </div>
          </button>
        </div>
        {isLoading && <p className="mt-8 text-blue-500 font-bold animate-pulse">Initializing Session...</p>}
      </div>
    );
  }

  return (
    <div className="w-[600px] min-h-[820px] bg-white rounded-[3rem] shadow-2xl overflow-hidden flex flex-col border-8 border-white relative">
      {/* Updated Task Header - Larger font and pulsing icon */}
      <div className="bg-slate-900 p-6 text-white flex justify-between items-center px-10">
        <div className="flex items-center gap-4">
            <div className="w-3 h-3 rounded-full bg-blue-400 animate-pulse shadow-[0_0_10px_rgba(96,165,250,0.6)]"></div>
            <span className="uppercase tracking-[0.2em] text-xs font-black opacity-80">{currentTrial?.phase}</span>
        </div>
        <div className="flex items-center gap-6">
            {currentTrial?.phase === 'learning' && session.condition === 'B' && localAttempt > 1 && (
                <div className="bg-orange-400 text-white text-[10px] font-black px-4 py-1 rounded-full animate-bounce uppercase tracking-widest">Attempt {localAttempt} / 3</div>
            )}
            <span className="text-xs font-mono opacity-60 uppercase tracking-widest">Task {currentTrial ? currentTrial.index + 1 : 0} OF {currentTrial?.total_in_phase}</span>
        </div>
      </div>

      <div className="p-10 flex flex-col flex-1">
        {/* Visual Prompt */}
        <div className="w-full aspect-video bg-slate-50 rounded-[2rem] mb-8 relative overflow-hidden flex items-center justify-center border-2 border-slate-100 group">
          {currentTrial && (
            <img 
              src={currentTrial.image_url} 
              alt="Vocabulary Target" 
              className="max-w-[75%] max-h-[75%] object-contain transition-transform group-hover:scale-105 duration-500"
              onError={(e: any) => { e.target.src = "https://via.placeholder.com/400x300?text=Image+Missing"; }}
            />
          )}
          {isLoading && <div className="absolute inset-0 bg-white/50 flex items-center justify-center"><p className="font-bold text-blue-500">Loading...</p></div>}
        </div>

        {/* Question Area */}
        <div className="text-center mb-6">
          <h2 className="text-2xl font-black text-slate-700 mb-1">
            {currentTrial?.task_type === 'article_mcq' ? 'Which article fits?' : 
             currentTrial?.task_type === 'plural_mcq' ? 'What is the plural form?' : 
             'What is the German name?'}
          </h2>
          <p className="text-lg text-slate-400 italic font-medium">"{currentTrial?.english_gloss}"</p>
          {currentTrial?.german_word && (
            <p className="mt-4 text-4xl font-black text-blue-500 tracking-tight">
              {currentTrial.german_word}
            </p>
          )}
        </div>

        {/* Input/Interaction Interface */}
        <div className="mt-auto">
          {!feedback?.move_next && currentTrial && (
            <div className="w-full">
              {currentTrial.task_type === 'article_mcq' || currentTrial.task_type === 'plural_mcq' ? (
                <div className="grid grid-cols-1 gap-3">
                  {currentTrial.options?.map(opt => (
                    <button
                      key={opt}
                      onClick={() => submitAnswer(opt)}
                      disabled={isLoading}
                      className="py-5 px-8 bg-slate-50 border-2 border-slate-100 rounded-2xl hover:bg-blue-600 hover:text-white hover:border-blue-600 transition-all font-bold text-slate-600 text-xl shadow-sm active:scale-95 disabled:opacity-50"
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col gap-4">
                  {/* Article Selection Buttons - Improved Visibility */}
                  <div className="flex gap-2 justify-center">
                    {['der', 'die', 'das'].map(art => (
                      <button
                        key={art}
                        onClick={() => { setSelectedArticle(art); setError(null); }}
                        className={`flex-1 py-3 rounded-xl font-black text-[10px] uppercase tracking-[0.2em] transition-all border-2 ${
                          selectedArticle === art 
                            ? 'bg-blue-600 text-white border-blue-700 shadow-lg' 
                            : 'bg-white text-slate-600 border-slate-400 hover:border-slate-500 shadow-sm'
                        }`}
                      >
                        {art}
                      </button>
                    ))}
                  </div>

                  {/* Noun Word Input */}
                  <div className="flex gap-3">
                    <input
                      autoFocus
                      type="text"
                      value={userInput}
                      onChange={(e) => { setUserInput(e.target.value); setError(null); }}
                      onKeyPress={(e) => e.key === 'Enter' && submitAnswer()}
                      className="flex-1 p-5 bg-blue-50/50 border-4 border-slate-100 rounded-2xl focus:border-blue-500 outline-none text-xl font-bold text-slate-700 placeholder-slate-300 shadow-inner"
                      placeholder="Type the noun..."
                    />
                    <button 
                        onClick={() => submitAnswer()} 
                        disabled={isLoading || !userInput.trim() || !selectedArticle} 
                        className="px-10 py-2 bg-blue-600 text-white rounded-2xl font-black text-xl shadow-lg hover:bg-blue-700 transition-all active:scale-95 disabled:bg-slate-100"
                    >
                        SEND
                    </button>
                  </div>
                  <GermanCharPicker />
                </div>
              )}
            </div>
          )}

          {/* Result Area */}
          {feedback && (
            <div className={`mt-6 p-8 rounded-[2rem] border-4 transition-all duration-500 ${
                feedback.is_correct ? 'bg-green-50 border-green-100 shadow-green-50' : 'bg-orange-50 border-orange-100 shadow-orange-50'
            } shadow-lg`}>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">{feedback.is_correct ? '✅' : (feedback.move_next ? 'ℹ️' : '💡')}</span>
                <p className="text-sm font-black text-slate-600 uppercase tracking-widest">
                  {feedback.is_correct ? 'Excellent!' : (feedback.move_next ? 'Solution' : 'Hint')}
                </p>
              </div>
              
              <p className="text-slate-600 text-lg font-medium leading-relaxed">
                {feedback.feedback}
              </p>

              {feedback.example && (
                <div className="mt-4 p-4 bg-white/60 rounded-xl border-2 border-green-50 shadow-sm">
                   <p className="text-slate-500 italic text-base font-semibold leading-relaxed">"{feedback.example}"</p>
                </div>
              )}
              
              {feedback.move_next && (
                <div className="mt-5 h-1.5 bg-slate-100 w-full rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 animate-[progress_2.5s_linear]"></div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Absolute Error Notification */}
      {error && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 w-full max-w-sm px-6 py-3 bg-red-600 text-white text-xs font-bold rounded-full shadow-2xl text-center flex items-center justify-center gap-2 z-[100]">
          <Icon.Info size={16} /> {error}
        </div>
      )}

      <style>{`
        @keyframes progress {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </div>
  );
};

export default ImageLabeling;