import React, { useState, useEffect, useCallback } from 'react';

// --- Icons (Inline SVG Components) ---
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
  FastForward: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="m13 19 9-7-9-7v14z"/><path d="m2 19 9-7-9-7v14z"/></svg>
  ),
  Info: ({ size = 24, className = "" }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
  )
};

// --- Interfaces ---

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

interface SessionData {
  session_id: string;
  condition: 'A' | 'B';
}

const ImageLabeling: React.FC = () => {
  const [session, setSession] = useState<SessionData | null>(null);
  const [currentTrial, setCurrentTrial] = useState<TrialData | null>(null);
  const [userInput, setUserInput] = useState<string>('');
  const [selectedArticle, setSelectedArticle] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<any>(null);
  const [startTime, setStartTime] = useState<number>(0);
  const [isComplete, setIsComplete] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showTransition, setShowTransition] = useState<boolean>(false);
  const [nextPhaseName, setNextPhaseName] = useState<string>('');
  const [localAttempt, setLocalAttempt] = useState<number>(1);
  const [mistakeHistory, setMistakeHistory] = useState<string[]>([]);

  useEffect(() => {
    document.body.style.backgroundColor = '#f8fafc';
    document.body.style.display = 'flex';
    document.body.style.alignItems = 'center';
    document.body.style.justifyContent = 'center';
    document.body.style.minHeight = '100vh';
    document.body.style.margin = '0';
  }, []);

  const startExperiment = async (condition: 'A' | 'B') => {
    setIsLoading(true);
    try {
      const resp = await fetch('/experiment/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: `user_${Date.now()}`, condition }),
      });
      const data = await resp.json();
      setSession(data);
      fetchNextTrial(data.session_id);
    } catch { setError("Connection error."); }
    finally { setIsLoading(false); }
  };

  const fetchNextTrial = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setFeedback(null);
    setUserInput('');
    setSelectedArticle(null);
    setLocalAttempt(1);
    setMistakeHistory([]);
    try {
      const resp = await fetch(`/experiment/trial/${sessionId}`);
      const data = await resp.json();
      if (data.status === "completed") setIsComplete(true);
      else if (currentTrial && currentTrial.phase !== data.phase) {
        setNextPhaseName(data.phase);
        setShowTransition(true);
        setCurrentTrial(data);
      } else {
        setCurrentTrial(data);
        setStartTime(Date.now() / 1000);
      }
    } finally { setIsLoading(false); }
  }, [currentTrial]);

  const skipToPhase = async (phase: string) => {
    if (!session) return;
    setIsLoading(true);
    try {
      await fetch('/experiment/skip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session.session_id, phase }),
      });
      fetchNextTrial(session.session_id);
    } catch { setError("Skip failed."); }
    finally { setIsLoading(false); }
  };

  const submitAnswer = async (answer?: string) => {
    let finalAnswer = answer || userInput;
    if (currentTrial?.task_type === 'type_word' && !answer) {
        if (!selectedArticle) { setError("Please select an article."); return; }
        if (!userInput.trim()) return;
        
        // Capitalization validation for Learning Condition B
        if (currentTrial?.phase === 'learning' && session?.condition === 'B' && userInput.trim()[0] !== userInput.trim()[0].toUpperCase()) {
            setFeedback({ is_correct: false, feedback: "Grammar Hint: In German, all nouns must be capitalized!", move_next: false });
            return;
        }
        finalAnswer = `${selectedArticle} ${userInput.trim()}`;
    }

    setIsLoading(true);
    setError(null);
    try {
      const resp = await fetch('/experiment/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          session_id: session?.session_id, 
          user_answer: finalAnswer, 
          start_time: startTime, 
          history: mistakeHistory 
        }),
      });
      const data = await resp.json();
      setFeedback(data);
      if (!data.move_next) {
          setLocalAttempt(prev => prev + 1);
          setMistakeHistory(prev => [...prev, finalAnswer]);
      }
    } finally { setIsLoading(false); }
  };

  const formatPhase = (p: string) => {
    switch(p) {
        case 'learning': return 'Learning Phase';
        case 'pre-test': return 'Pre-Test';
        case 'post-test': return 'Post-Test';
        default: return p.charAt(0).toUpperCase() + p.slice(1);
    }
  };

  if (isComplete) return (
    <div className="w-full max-w-xl bg-white rounded-[2.5rem] shadow-2xl text-center p-12 border-8 border-white">
      <Icon.CheckCircle size={64} className="text-green-500 mx-auto mb-6" />
      <h1 className="text-4xl font-black text-slate-700 mb-8 tracking-tight">Experiment Finished!</h1>
      <button onClick={() => window.location.reload()} className="px-10 py-4 bg-blue-600 text-white rounded-2xl font-bold shadow-lg hover:bg-blue-700 active:scale-95 transition-all">Restart</button>
    </div>
  );

  if (showTransition) return (
    <div className="w-full max-w-xl bg-white rounded-[2.5rem] shadow-2xl text-center p-12 border-8 border-white">
      {nextPhaseName === 'learning' ? <Icon.BookOpen size={64} className="text-blue-500 mx-auto mb-6" /> : <Icon.GraduationCap size={64} className="text-purple-600 mx-auto mb-6" />}
      <h2 className="text-3xl font-black text-slate-700 mb-8 capitalize">Starting {formatPhase(nextPhaseName)}</h2>
      <button onClick={() => { setShowTransition(false); setStartTime(Date.now() / 1000); }} className="px-10 py-4 bg-slate-800 text-white rounded-2xl font-bold shadow-xl flex items-center gap-2 mx-auto hover:bg-slate-700 transition-all">Start <Icon.FastForward size={20} /></button>
    </div>
  );

  if (!session) return (
    <div className="w-full max-w-3xl text-center">
      <h1 className="text-5xl font-black mb-12 text-slate-800 tracking-tight">Vocabulary Study</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 px-4">
        {['A', 'B'].map(cond => (
          <button key={cond} onClick={() => startExperiment(cond as any)} className="p-10 bg-white border-4 border-transparent hover:border-blue-400 rounded-[3rem] shadow-xl text-left flex items-start gap-6 group transition-all h-full">
            <div className={`p-5 rounded-3xl transition-colors ${cond === 'A' ? 'bg-blue-50 text-blue-400 group-hover:bg-blue-600 group-hover:text-white' : 'bg-purple-50 text-purple-400 group-hover:bg-purple-600 group-hover:text-white'}`}>{cond === 'A' ? <Icon.XCircle size={40} /> : <Icon.CheckCircle size={40} />}</div>
            <div><h2 className="text-2xl font-bold text-slate-700 mb-1">Group {cond}</h2><p className="text-slate-500">{cond === 'A' ? 'Static' : 'Adaptive AI'} Feedback</p></div>
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-4xl px-4 py-8">
      <div className="w-full bg-white rounded-[2.5rem] shadow-2xl overflow-hidden flex flex-col border-8 border-white relative">
        {/* Header */}
        <div className="bg-slate-900 p-4 text-white flex justify-between items-center px-8">
          <div className="flex items-center gap-3"><div className="w-2.5 h-2.5 rounded-full bg-blue-400 animate-pulse shadow-[0_0_8px_rgba(96,165,250,0.6)]"></div><span className="uppercase tracking-widest text-[10px] font-black opacity-80">{formatPhase(currentTrial?.phase || '')}</span></div>
          <div className="flex items-center gap-4">
              {currentTrial?.phase === 'learning' && session.condition === 'B' && localAttempt > 1 && !feedback?.move_next && <div className="bg-orange-400 text-white text-[9px] font-black px-3 py-1 rounded-full animate-pulse uppercase tracking-widest">Attempt {localAttempt}/3</div>}
              <span className="text-[10px] font-mono opacity-60 uppercase tracking-widest">Item {currentTrial ? currentTrial.index + 1 : 0} of {currentTrial?.total_in_phase}</span>
          </div>
        </div>

        <div className="p-8 flex flex-col gap-6 flex-1">
          {/* Side-by-Side Main Content */}
          <div className="flex flex-col lg:flex-row gap-8 items-stretch">
              {/* Left: Image */}
              <div className="flex-1 min-h-[300px] bg-slate-50 rounded-[1.5rem] relative flex items-center justify-center border-2 border-slate-100 group overflow-hidden">
                  <img src={currentTrial?.image_url} alt="Vocab" className="max-w-[80%] max-h-[80%] object-contain transition-transform group-hover:scale-105 duration-500" onError={(e:any)=>e.target.src="https://via.placeholder.com/200?text=Missing+Image"} />
                  {isLoading && !feedback && <div className="absolute inset-0 bg-white/50 flex items-center justify-center z-10 font-bold text-blue-500">Loading...</div>}
              </div>

              {/* Right: Question and Interaction */}
              <div className="flex-1 flex flex-col justify-center gap-5">
                  <div className="space-y-1 text-center lg:text-left">
                    <h2 className="text-2xl font-black text-slate-700 leading-tight">{currentTrial?.task_type === 'article_mcq' ? 'Welcher Artikel passt?' : currentTrial?.task_type === 'plural_mcq' ? 'Wie lautet die Pluralform?' : 'Wie heißt das auf Deutsch?'}</h2>
                    <p className="text-xl text-slate-400 italic font-medium">"{currentTrial?.english_gloss}"</p>
                    {currentTrial?.german_word && <p className="mt-3 text-4xl font-black text-blue-600 tracking-tight">{currentTrial.german_word}</p>}
                  </div>

                  <div className="w-full mt-2">
                      {currentTrial?.task_type !== 'type_word' ? (
                          <div className="grid grid-cols-1 gap-2.5">
                            {currentTrial?.options?.map((opt:string) => (
                              <button key={opt} onClick={() => submitAnswer(opt)} disabled={isLoading || feedback?.move_next} className={`py-4 px-6 border-2 rounded-xl font-bold text-lg shadow-sm transition-all active:scale-95 text-left flex justify-between items-center ${feedback?.move_next ? 'bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed' : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-blue-600 hover:text-white hover:border-blue-600'}`}>
                                {opt}
                                {!feedback?.move_next && <Icon.ArrowRight size={18} className="opacity-0 group-hover:opacity-100" />}
                              </button>
                            ))}
                          </div>
                      ) : (
                          <div className="flex flex-col gap-3">
                              {/* Article Buttons */}
                              <div className="flex gap-2.5 justify-center">
                                {['der', 'die', 'das'].map(art => (
                                  <button key={art} onClick={() => { setSelectedArticle(art); setError(null); }} disabled={feedback?.move_next} className={`flex-1 py-3.5 rounded-xl font-black text-[9px] uppercase tracking-[0.15em] transition-all border-2 ${selectedArticle === art ? 'bg-blue-600 text-white border-blue-700 shadow-lg' : 'bg-white text-slate-600 border-slate-400 hover:border-slate-500 shadow-sm'} ${feedback?.move_next ? 'opacity-50 cursor-not-allowed' : ''}`}>{art}</button>
                                ))}
                              </div>
                              {/* Input and Send */}
                              <div className="flex gap-2">
                                <input autoFocus type="text" value={userInput} onChange={(e) => { setUserInput(e.target.value); setError(null); }} onKeyPress={(e) => e.key === 'Enter' && submitAnswer()} disabled={isLoading || feedback?.move_next} className={`flex-1 p-4 border-2 rounded-xl outline-none text-xl font-bold shadow-inner transition-colors ${feedback?.move_next ? 'bg-slate-50 border-slate-100 text-slate-300' : 'bg-blue-50/50 border-slate-100 focus:border-blue-500 text-slate-700 placeholder-slate-300'}`} placeholder="Type noun..." />
                                {!feedback?.move_next && <button onClick={() => submitAnswer()} disabled={isLoading || !userInput.trim() || !selectedArticle} className="px-8 py-2 bg-blue-600 text-white rounded-xl font-black text-lg active:scale-95 shadow-lg hover:bg-blue-700 transition-all disabled:bg-slate-100">SEND</button>}
                              </div>
                              {/* Char Picker */}
                              <div className="flex gap-2 mt-1">
                                {['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'].map(c => (
                                  <button key={c} onClick={()=>setUserInput(p=>p+c)} disabled={feedback?.move_next} className={`w-9 h-9 bg-slate-50 border border-slate-300 rounded font-bold text-slate-500 text-sm transition-all hover:bg-white hover:border-blue-400 shadow-sm ${feedback?.move_next ? 'opacity-50' : ''}`}>{c}</button>
                                ))}
                              </div>
                          </div>
                      )}
                  </div>
              </div>
          </div>

          {/* Feedback & Navigation */}
          {feedback && (
              <div className="flex flex-col gap-5 pt-2 border-t border-slate-100 animate-in fade-in duration-500">
                  <div className={`p-6 rounded-[1.5rem] border-4 shadow-md transition-all ${feedback.is_correct ? 'bg-green-50 border-green-100' : (currentTrial.phase !== 'learning' ? 'bg-red-50 border-red-100' : 'bg-orange-50 border-orange-100')}`}>
                      <div className="flex items-center gap-3 mb-3">
                          <div className={`p-2 rounded-xl ${feedback.is_correct ? 'bg-green-100 text-green-600' : (currentTrial.phase !== 'learning' ? 'bg-red-100 text-red-600' : (feedback.move_next ? 'bg-blue-100 text-blue-600' : 'bg-orange-100 text-orange-600'))}`}>
                            {feedback.is_correct ? <Icon.CheckCircle size={24} /> : (currentTrial.phase !== 'learning' ? <Icon.XCircle size={24} /> : (feedback.move_next ? <Icon.Info size={24} /> : <Icon.PlayCircle size={24} />))}
                          </div>
                          <p className="text-xl font-black text-slate-700 uppercase tracking-widest">{feedback.is_correct ? 'Correct!' : (currentTrial.phase !== 'learning' ? 'Wrong' : (feedback.move_next ? 'Correction Tip' : 'Hint'))}</p>
                      </div>
                      <div className="space-y-4">
                          {currentTrial.phase !== 'learning' && !feedback.is_correct ? (
                            <p className="text-slate-700 text-xl font-medium leading-relaxed">The correct answer is: <span className="underline font-black text-red-700">{feedback.feedback}</span></p>
                          ) : (
                            <div className="space-y-4">
                                {feedback.move_next && !feedback.is_correct && currentTrial.phase === 'learning' && session.condition === 'B' ? (
                                    <div className="bg-white/40 p-4 rounded-2xl border-l-8 border-blue-500">
                                        <p className="text-blue-700 font-bold text-lg uppercase tracking-tight mb-2">Personalized Correction Tip:</p>
                                        <p className="text-slate-700 text-lg font-medium leading-relaxed italic">{feedback.feedback}</p>
                                    </div>
                                ) : (
                                    <p className="text-slate-700 text-lg font-medium leading-relaxed">{feedback.feedback}</p>
                                )}
                            </div>
                          )}
                          {feedback.example && <div className="mt-4 p-4 bg-white/60 rounded-xl border-2 border-green-50 shadow-sm text-slate-500 italic font-semibold leading-relaxed">"{feedback.example}"</div>}
                      </div>
                  </div>
                  {feedback.move_next && <button onClick={() => fetchNextTrial(session.session_id)} className="w-full py-5 bg-slate-900 text-white rounded-[2rem] font-black text-xl shadow-xl flex items-center justify-center gap-3 hover:bg-slate-800 transition-all active:scale-[0.98] group">NEXT TASK <Icon.ArrowRight size={24} className="group-hover:translate-x-1.5 transition-transform" /></button>}
              </div>
          )}
        </div>
      </div>

      {/* Testing Toolbar */}
      <div className="w-full bg-slate-800 p-4 rounded-3xl flex flex-wrap items-center justify-center gap-4 border-4 border-slate-700 shadow-xl mt-4">
        <span className="text-slate-400 font-bold text-xs uppercase tracking-widest mr-2 flex items-center gap-2">
          <Icon.Info size={14} className="text-blue-400" /> Testing Controls:
        </span>
        <button onClick={() => skipToPhase('pre-test')} className="px-4 py-2 bg-slate-700 text-slate-200 rounded-xl text-[10px] font-black hover:bg-blue-600 hover:text-white transition-all uppercase tracking-widest">Pre-Test</button>
        <button onClick={() => skipToPhase('learning')} className="px-4 py-2 bg-slate-700 text-slate-200 rounded-xl text-[10px] font-black hover:bg-blue-600 hover:text-white transition-all uppercase tracking-widest">Learning</button>
        <button onClick={() => skipToPhase('post-test')} className="px-4 py-2 bg-slate-700 text-slate-200 rounded-xl text-[10px] font-black hover:bg-blue-600 hover:text-white transition-all uppercase tracking-widest">Post-Test</button>
      </div>

      {error && <div className="absolute top-4 left-1/2 -translate-x-1/2 w-full max-w-sm px-5 py-3 bg-red-600 text-white text-xs font-bold rounded-full shadow-2xl flex items-center justify-center gap-2 z-[100] animate-bounce"><Icon.Info size={16} /> {error}</div>}
    </div>
  );
};

export default ImageLabeling;