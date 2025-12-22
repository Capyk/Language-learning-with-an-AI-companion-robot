import React, { useState, useEffect, useCallback } from 'react';

// --- Icons (Inline SVG Components to avoid lucide-react dependency) ---
const Icon = {
  Send: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
  ),
  CheckCircle: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
  ),
  XCircle: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
  ),
  ArrowRight: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
  ),
  BookOpen: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
  ),
  GraduationCap: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
  ),
  PlayCircle: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
  ),
  Info: ({ size = 24, className = "" }) => (
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
  const [session, setSession] = useState<SessionData | null>(null);
  const [currentTrial, setCurrentTrial] = useState<TrialData | null>(null);
  const [userInput, setUserInput] = useState<string>('');
  const [selectedArticle, setSelectedArticle] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<FeedbackData | null>(null);
  const [startTime, setStartTime] = useState<number>(0);
  const [isComplete, setIsComplete] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showTransition, setShowTransition] = useState<boolean>(false);
  const [nextPhaseName, setNextPhaseName] = useState<string>('');
  const [localAttempt, setLocalAttempt] = useState<number>(1);

  // Styling effect for centering on web
  useEffect(() => {
    document.body.style.backgroundColor = '#f8fafc';
    document.body.style.margin = '0';
    document.body.style.display = 'flex';
    document.body.style.alignItems = 'center';
    document.body.style.justifyContent = 'center';
    document.body.style.minHeight = '100vh';
    return () => { document.body.style.backgroundColor = ''; };
  }, []);

  // 1. Initialize Experiment
  const startExperiment = async (condition: 'A' | 'B') => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/experiment/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          user_id: `user_${Math.floor(Math.random() * 1000)}`,
          condition: condition 
        }),
      });
      if (!response.ok) throw new Error("Failed to initialize session");
      const data: SessionData = await response.json();
      setSession(data);
      fetchNextTrial(data.session_id);
    } catch (err) {
      setError("Connection to backend server failed.");
    } finally {
      setIsLoading(false);
    }
  };

  // 2. Fetch Next Trial
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
      setError("Error loading the next task.");
    } finally {
      setIsLoading(false);
    }
  }, [currentTrial]);

  // 3. Submit Answer
  const submitAnswer = async (answer?: string) => {
    let finalAnswer = "";
    
    if (currentTrial?.task_type === 'type_word') {
        if (!selectedArticle) {
            setError("Please select an article (der/die/das) first.");
            return;
        }
        if (!userInput.trim()) return;

        const trimmedInput = userInput.trim();
        if (currentTrial?.phase === 'learning' && session?.condition === 'B') {
          if (trimmedInput[0] !== trimmedInput[0].toUpperCase()) {
              setFeedback({
                  is_correct: false,
                  feedback: "Grammar Hint: In German, all nouns must be capitalized!",
                  move_next: false
              });
              return;
          }
        }
        finalAnswer = `${selectedArticle} ${trimmedInput}`;
    } else {
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

      if (!data.move_next) {
        setLocalAttempt(prev => prev + 1);
      }
    } catch (err) {
      setError("Submission failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const insertChar = (char: string) => {
    if (feedback?.move_next) return;
    setUserInput(prev => prev + char);
  };

  const GermanCharPicker = () => (
    <div className="flex gap-2 mt-2">
      {['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'].map(char => (
        <button
          key={char}
          onClick={() => insertChar(char)}
          disabled={feedback?.move_next}
          className={`w-9 h-9 flex items-center justify-center bg-slate-50 border border-slate-300 rounded hover:bg-white hover:border-blue-400 font-bold text-slate-500 text-sm transition-all shadow-sm ${feedback?.move_next ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {char}
        </button>
      ))}
    </div>
  );

  if (isComplete) {
    return (
      <div className="w-full max-w-xl min-h-[350px] flex flex-col items-center justify-center bg-white rounded-[2.5rem] shadow-2xl text-center p-10 border-8 border-white">
        <div className="bg-green-50 p-5 rounded-full mb-5">
          <Icon.CheckCircle size={48} className="text-green-500" />
        </div>
        <h1 className="text-3xl font-black text-slate-700 mb-3 tracking-tight">All Done!</h1>
        <p className="text-slate-500 text-base mb-6 leading-relaxed">
          The experiment is complete. Thank you for participating!
        </p>
        <button 
          onClick={() => window.location.reload()}
          className="px-8 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 transition-all shadow-lg active:scale-95"
        >
          Restart
        </button>
      </div>
    );
  }

  const formatPhaseName = (p: string) => {
      switch(p) {
          case 'learning': return 'Learning Phase';
          case 'pre-test': return 'Pre-Test';
          case 'post-test': return 'Post-Test';
          default: return p.charAt(0).toUpperCase() + p.slice(1);
      }
  };

  if (showTransition) {
    return (
      <div className="w-full max-w-xl min-h-[350px] flex flex-col items-center justify-center bg-white rounded-[2.5rem] shadow-2xl text-center p-10 border-8 border-white">
        <div className="bg-blue-50 p-5 rounded-full mb-5">
          {nextPhaseName === 'learning' ? (
            <Icon.BookOpen size={48} className="text-blue-500" />
          ) : (
            <Icon.GraduationCap size={48} className="text-purple-600" />
          )}
        </div>
        <h2 className="text-2xl font-black text-slate-700 mb-3">
          Starting {formatPhaseName(nextPhaseName)}
        </h2>
        <p className="text-slate-500 text-base mb-8 leading-relaxed px-4">
            {nextPhaseName === 'learning' 
                ? "Time to learn! In this phase, you will receive tips and hints if you make a mistake." 
                : "Great job! Let's see how much you remember in the final assessment."}
        </p>
        <button 
          onClick={() => {
            setShowTransition(false);
            setStartTime(Date.now() / 1000);
          }}
          className="flex items-center gap-2 px-8 py-3 bg-slate-800 text-white rounded-xl font-bold hover:bg-slate-700 transition-all active:scale-95 shadow-xl"
        >
          Start Phase <Icon.PlayCircle size={18} />
        </button>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="w-full max-w-3xl text-center flex flex-col items-center p-4">
        <h1 className="text-4xl font-black mb-3 text-slate-800 tracking-tight">Vocabulary Study</h1>
        <p className="text-slate-500 mb-10 text-lg">Please select a condition to begin.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full px-4">
          <button 
            onClick={() => startExperiment('A')}
            className="p-8 bg-white border-4 border-transparent hover:border-blue-400 rounded-[2rem] shadow-xl transition-all group text-left flex items-start gap-5 h-full"
          >
            <div className="bg-blue-50 p-4 rounded-2xl text-blue-400 group-hover:bg-blue-600 group-hover:text-white transition-colors shrink-0">
              <Icon.XCircle size={32} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-700 mb-1">Group A</h2>
              <p className="text-slate-500 text-sm">Static Feedback (Instant correction only).</p>
            </div>
          </button>
          <button 
            onClick={() => startExperiment('B')}
            className="p-8 bg-white border-4 border-transparent hover:border-purple-400 rounded-[2rem] shadow-xl transition-all group text-left flex items-start gap-5 h-full"
          >
            <div className="bg-purple-50 p-4 rounded-2xl text-purple-600 group-hover:bg-purple-600 group-hover:text-white transition-colors shrink-0">
              <Icon.CheckCircle size={32} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-700 mb-1">Group B</h2>
              <p className="text-slate-500 text-sm">Adaptive Feedback (AI scaffolding hints).</p>
            </div>
          </button>
        </div>
        {isLoading && <p className="mt-10 text-blue-500 text-lg font-bold animate-pulse">Initializing session...</p>}
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl min-h-[600px] bg-white rounded-[2.5rem] shadow-2xl overflow-hidden flex flex-col border-8 border-white relative mx-4 my-8">
      {/* Header Info Bar */}
      <div className="bg-slate-900 p-4 text-white flex justify-between items-center px-8">
        <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-blue-400 animate-pulse shadow-[0_0_8px_rgba(96,165,250,0.6)]"></div>
            <span className="uppercase tracking-[0.15em] text-[10px] font-black opacity-80">{formatPhaseName(currentTrial?.phase || '')}</span>
        </div>
        <div className="flex items-center gap-4">
            {currentTrial?.phase === 'learning' && session.condition === 'B' && localAttempt > 1 && !feedback?.move_next && (
                <div className="bg-orange-400 text-white text-[9px] font-black px-3 py-1 rounded-full animate-pulse uppercase tracking-widest">Attempt {localAttempt} / 3</div>
            )}
            <span className="text-[10px] font-mono opacity-60 uppercase tracking-widest">Task {currentTrial ? currentTrial.index + 1 : 0} of {currentTrial?.total_in_phase}</span>
        </div>
      </div>

      <div className="p-8 flex flex-col gap-6 flex-1">
        
        {/* Main Content Area: Side-by-Side */}
        <div className="flex flex-col lg:flex-row gap-8 items-stretch">
            
            {/* Left Column: Image area */}
            <div className="flex-1 min-h-[280px] bg-slate-50 rounded-[1.5rem] relative overflow-hidden flex items-center justify-center border-2 border-slate-100 group">
                {currentTrial && (
                    <img 
                    src={currentTrial.image_url} 
                    alt="Task Visual" 
                    className="max-w-[80%] max-h-[80%] object-contain transition-transform group-hover:scale-105 duration-500"
                    onError={(e: any) => { e.target.src = "https://via.placeholder.com/400x300?text=Image+Missing"; }}
                    />
                )}
                {isLoading && !feedback && <div className="absolute inset-0 bg-white/50 flex items-center justify-center z-10"><p className="font-bold text-blue-500 text-sm">Loading...</p></div>}
            </div>

            {/* Right Column: Question & Interaction Area */}
            <div className="flex-1 flex flex-col justify-center gap-5">
                <div className="space-y-1">
                    <h2 className="text-2xl font-black text-slate-700 leading-tight">
                        {currentTrial?.task_type === 'article_mcq' ? 'Welcher Artikel passt?' : 
                        currentTrial?.task_type === 'plural_mcq' ? 'Wie lautet die Pluralform?' : 
                        'Wie heißt das auf Deutsch?'}
                    </h2>
                    <p className="text-xl text-slate-400 italic font-medium">"{currentTrial?.english_gloss}"</p>
                    
                    {currentTrial?.german_word && (
                        <p className="mt-3 text-4xl font-black text-blue-600 tracking-tight">
                        {currentTrial.german_word}
                        </p>
                    )}
                </div>

                {/* Interaction - Buttons or Input */}
                <div className="w-full mt-2">
                    {currentTrial?.task_type === 'article_mcq' || currentTrial?.task_type === 'plural_mcq' ? (
                    <div className="grid grid-cols-1 gap-2.5">
                        {currentTrial.options?.map(opt => (
                        <button
                            key={opt}
                            onClick={() => submitAnswer(opt)}
                            disabled={isLoading || feedback?.move_next}
                            className={`py-4 px-6 border-2 rounded-xl font-bold text-lg shadow-sm transition-all active:scale-95 text-left flex justify-between items-center ${
                            feedback?.move_next 
                                ? 'bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed'
                                : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-blue-600 hover:text-white hover:border-blue-600'
                            }`}
                        >
                            {opt}
                            {!feedback?.move_next && <Icon.ArrowRight size={18} className="opacity-0 group-hover:opacity-100" />}
                        </button>
                        ))}
                    </div>
                    ) : (
                    <div className="flex flex-col gap-3">
                        {/* Article Selection Buttons */}
                        <div className="flex gap-2.5 justify-center">
                        {['der', 'die', 'das'].map(art => (
                            <button
                            key={art}
                            onClick={() => { setSelectedArticle(art); setError(null); }}
                            disabled={feedback?.move_next}
                            className={`flex-1 py-3.5 rounded-xl font-black text-[9px] uppercase tracking-[0.15em] transition-all border-2 ${
                                selectedArticle === art 
                                ? 'bg-blue-600 text-white border-blue-700 shadow-lg' 
                                : 'bg-white text-slate-600 border-slate-400 hover:border-slate-500 shadow-sm'
                            } ${feedback?.move_next ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                            {art}
                            </button>
                        ))}
                        </div>

                        {/* Noun Word Input */}
                        <div className="flex gap-2.5">
                        <input
                            autoFocus
                            type="text"
                            value={userInput}
                            onChange={(e) => { setUserInput(e.target.value); setError(null); }}
                            onKeyPress={(e) => e.key === 'Enter' && submitAnswer()}
                            disabled={isLoading || feedback?.move_next}
                            className={`flex-1 p-4 border-2 rounded-xl outline-none text-xl font-bold shadow-inner transition-colors ${
                            feedback?.move_next 
                                ? 'bg-slate-50 border-slate-100 text-slate-300' 
                                : 'bg-blue-50/50 border-slate-100 focus:border-blue-500 text-slate-700 placeholder-slate-300'
                            }`}
                            placeholder="Type the noun..."
                        />
                        {!feedback?.move_next && (
                            <button 
                                onClick={() => submitAnswer()} 
                                disabled={isLoading || !userInput.trim() || !selectedArticle} 
                                className="px-8 py-2 bg-blue-600 text-white rounded-xl font-black text-lg shadow-lg hover:bg-blue-700 transition-all active:scale-95 disabled:bg-slate-100"
                            >
                                SEND
                            </button>
                        )}
                        </div>
                        <GermanCharPicker />
                    </div>
                    )}
                </div>
            </div>
        </div>

        {/* Bottom Section: Feedback and Next Task */}
        {feedback && (
            <div className="flex flex-col gap-5 pt-2 border-t border-slate-100">
                <div className={`p-6 rounded-[1.5rem] border-4 transition-all duration-500 ${
                    feedback.is_correct ? 'bg-green-50 border-green-100 shadow-green-50' : 'bg-orange-50 border-orange-100 shadow-orange-100'
                } shadow-md`}>
                    <div className="flex items-center gap-3 mb-3">
                        <div className={`p-2 rounded-xl ${feedback.is_correct ? 'bg-green-100 text-green-600' : 'bg-orange-100 text-orange-600'}`}>
                            {feedback.is_correct ? <Icon.CheckCircle size={24} /> : (feedback.move_next ? <Icon.Info size={24} /> : <Icon.PlayCircle size={24} />)}
                        </div>
                        <p className="text-xl font-black text-slate-700 uppercase tracking-widest">
                            {feedback.is_correct ? 'Correct!' : (feedback.move_next ? 'Solution' : 'Hint')}
                        </p>
                    </div>
                    
                    <p className="text-slate-600 text-lg font-medium leading-relaxed mb-3">
                        {feedback.feedback}
                    </p>

                    {feedback.example && (
                        <div className="mt-4 p-4 bg-white/60 rounded-xl border-2 border-green-50 shadow-sm">
                            <p className="text-slate-500 italic text-lg font-semibold leading-relaxed">"{feedback.example}"</p>
                        </div>
                    )}
                </div>

                {/* Manual Navigation - Web-style Button */}
                {feedback.move_next && (
                    <button
                        onClick={() => { if (session) fetchNextTrial(session.session_id); }}
                        className="w-full py-4 bg-slate-900 text-white rounded-2xl font-black text-xl shadow-xl flex items-center justify-center gap-3 hover:bg-slate-800 transition-all active:scale-[0.98] group"
                    >
                        NEXT TASK 
                        <Icon.ArrowRight size={24} className="group-hover:translate-x-1.5 transition-transform" />
                    </button>
                )}
            </div>
        )}

      </div>
      
      {/* Absolute Error Notification */}
      {error && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 w-full max-w-sm px-5 py-3 bg-red-600 text-white text-xs font-bold rounded-full shadow-2xl text-center flex items-center justify-center gap-2 z-[100]">
          <Icon.Info size={16} /> {error}
        </div>
      )}
    </div>
  );
};

export default ImageLabeling;