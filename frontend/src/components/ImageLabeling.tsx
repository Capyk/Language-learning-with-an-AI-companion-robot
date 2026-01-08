import React, { useState, useEffect, useCallback } from 'react';

// --- SZTYWNY ADRES BACKENDU (BEZ IF-ÓW, BEZ ZMIENNYCH) ---
// Dzięki temu frontend ZAWSZE wie, że ma pytać ten serwer, a nie siebie.
const API_BASE = 'https://german-learning-language-backend.onrender.com';

// --- ICONS ---
const Icon = {
  CheckCircle: ({ size = 24, className = "" }: any) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
  ),
  XCircle: ({ size = 24, className = "" }: any) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
  ),
  ArrowRight: ({ size = 24, className = "" }: any) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
  ),
  Lightbulb: ({ size = 24, className = "" }: any) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M15 14c.2-.8.8-1.5 1.7-2"/><circle cx="12" cy="12" r="10"/></svg>
  ),
  Info: ({ size = 24, className = "" }: any) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
  )
};

const formatText = (text: string) => {
    if (!text) return null;
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, index) => {
        if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={index} className="font-black text-indigo-900">{part.slice(2, -2)}</strong>;
        }
        return <span key={index}>{part}</span>;
    });
};

const DemographicsForm = ({ onSubmit }: { onSubmit: (data: any) => void }) => {
    const [formData, setFormData] = useState({ age: '', gender: '', education: '', german_level: '' });
    const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); onSubmit(formData); };

    return (
        <div className="w-full max-w-2xl bg-white rounded-[2.5rem] shadow-2xl p-10 border-8 border-white mx-auto animate-in fade-in zoom-in duration-500">
            <div className="text-center mb-8">
                <h1 className="text-3xl font-black text-slate-800 mb-2">Final Step</h1>
                <p className="text-slate-500">Please provide basic info to save results.</p>
            </div>
            <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                    <label className="block text-slate-700 font-bold mb-2 ml-1">Age</label>
                    <input type="number" required min="10" max="99" value={formData.age} onChange={e => setFormData({...formData, age: e.target.value})} className="w-full p-4 rounded-xl border-2 border-slate-200 text-slate-800 font-bold outline-none focus:border-blue-500 bg-white" placeholder="e.g. 25" />
                </div>
                <div>
                    <label className="block text-slate-700 font-bold mb-2 ml-1">Gender</label>
                    <div className="grid grid-cols-2 gap-4">
                        {['Male', 'Female', 'Other', 'Prefer not to say'].map(opt => (
                            <button type="button" key={opt} onClick={() => setFormData({...formData, gender: opt})} className={`p-4 rounded-xl border-2 font-bold transition-all ${formData.gender === opt ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'}`}>{opt}</button>
                        ))}
                    </div>
                </div>
                <div>
                    <label className="block text-slate-700 font-bold mb-2 ml-1">Education</label>
                    <select required value={formData.education} onChange={e => setFormData({...formData, education: e.target.value})} className="w-full p-4 rounded-xl border-2 border-slate-200 text-slate-800 font-bold outline-none focus:border-blue-500 bg-white appearance-none">
                        <option value="" disabled>Select option...</option>
                        <option value="High School">High School</option>
                        <option value="Bachelor">Bachelor's</option>
                        <option value="Master">Master's</option>
                        <option value="PhD">PhD</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
                <div>
                    <label className="block text-slate-700 font-bold mb-2 ml-1">German Level</label>
                    <div className="flex gap-2">
                        {['A0', 'A1', 'A2', 'B1+'].map(lvl => (
                            <button type="button" key={lvl} onClick={() => setFormData({...formData, german_level: lvl})} className={`flex-1 p-3 rounded-xl border-2 font-bold text-sm transition-all ${formData.german_level === lvl ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'}`}>{lvl}</button>
                        ))}
                    </div>
                </div>
                <button type="submit" disabled={!formData.age || !formData.gender || !formData.education || !formData.german_level} className="w-full py-5 bg-green-600 text-white rounded-2xl font-black text-xl hover:bg-green-700 transition-all shadow-lg active:scale-95 disabled:bg-slate-300 mt-4">SUBMIT & FINISH</button>
            </form>
        </div>
    );
};

const LearningScreenRenderer = ({ data, onNext }: { data: any, onNext: () => void }) => {
  const [localInput, setLocalInput] = useState("");
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'checked'>('idle');
  const [isCorrect, setIsCorrect] = useState(false);

  useEffect(() => {
    setLocalInput("");
    setSelectedOption(null);
    setStatus('idle');
    setIsCorrect(false);
  }, [data.step_number]);

  const getArticleColor = (art: string) => {
    if (!art) return 'text-slate-800 bg-slate-50';
    const lower = art.toLowerCase();
    if (lower === 'der') return 'text-blue-600 bg-blue-50 border-blue-200';
    if (lower === 'die') return 'text-red-600 bg-red-50 border-red-200';
    if (lower === 'das') return 'text-green-600 bg-green-50 border-green-200';
    return 'text-slate-800 bg-slate-50';
  };

  const handleCheck = () => {
      let correct = false;
      const target = data.german_word ? data.german_word.trim().toLowerCase() : "";

      if (data.interaction_type === 'fill_gap') {
          const input = localInput.trim().toLowerCase();
          correct = (input === target) || target.endsWith(" " + input);
      } else if (data.interaction_type === 'choice') {
          const selection = selectedOption?.toLowerCase() || "";
          correct = target.startsWith(selection + " ") || target === selection;
      }
      setIsCorrect(correct);
      setStatus('checked');
  };

  const renderContextWithGap = () => {
    let context = data.question_context || "_______";
    if (!context.includes('_______')) context += " _______";

    const parts = context.split('_______');
    const preText = parts[0] || "";
    const postText = parts.length > 1 ? parts[1] : "";

    return (
        <div className="flex flex-wrap items-center gap-2 text-xl font-mono bg-white p-4 rounded-xl shadow-sm leading-loose w-full border border-slate-200 justify-center">
            {preText && <span className="text-slate-700">{formatText(preText)}</span>}
            <div className="relative inline-block min-w-[140px]">
                <input 
                    type="text" 
                    value={localInput}
                    onChange={(e) => setLocalInput(e.target.value)}
                    className={`w-full p-2 text-center font-bold border-b-4 outline-none bg-transparent transition-colors text-slate-900 ${
                         status === 'checked'
                         ? (isCorrect ? 'border-green-500 text-green-700' : 'border-red-500 text-red-700')
                         : 'border-indigo-300 focus:border-indigo-600'
                    }`}
                    disabled={status === 'checked'}
                    autoFocus
                    placeholder="..."
                />
            </div>
            {postText && <span className="text-slate-700">{formatText(postText)}</span>}
        </div>
    );
  };

  const isFullWidth = ['story', 'intro', 'summary', 'fun_fact', 'dialogue'].includes(data.visual_type);
  const showImage = !isFullWidth && !!data.image_url;

  // --- BUDOWANIE URL OBRAZKA (ABSOLUTNA ŚCIEŻKA) ---
  // Jeśli backend zwrócił "/images/img_01.jpg", my doklejamy "https://....com" na początku.
  const imageUrl = data.image_url 
    ? (data.image_url.startsWith('http') ? data.image_url : `${API_BASE}${data.image_url}`)
    : null;

  return (
    <div className="w-full max-w-7xl bg-white rounded-[2.5rem] shadow-2xl overflow-hidden border-8 border-white mx-auto animate-in fade-in slide-in-from-bottom-8 duration-700 flex flex-col min-h-[600px]">
      
      <div className="bg-indigo-50 p-6 flex justify-between items-center border-b border-indigo-100 shrink-0">
        <span className="bg-white text-indigo-700 px-4 py-2 rounded-full text-sm font-black uppercase tracking-widest shadow-sm">
          Step {data.step_number}
        </span>
        {data.mnemonics && (
          <span className="flex items-center gap-2 text-amber-600 bg-amber-100 px-4 py-2 rounded-full text-xs font-black uppercase border border-amber-200">
            <Icon.Lightbulb size={16} /> AI Tip
          </span>
        )}
      </div>

      <div className={`flex-1 ${showImage ? 'grid grid-cols-2' : 'flex flex-col'}`}>
          {showImage ? (
              <div className="bg-slate-100 flex flex-col items-center justify-center p-8 border-r border-slate-200 h-full relative overflow-hidden">
                  {imageUrl ? (
                      <>
                        <img 
                            src={imageUrl} 
                            alt="visual" 
                            onError={(e) => { e.currentTarget.style.display='none'; }}
                            className="w-auto h-auto max-w-full max-h-[450px] object-contain drop-shadow-2xl rounded-2xl transition-transform hover:scale-105 duration-500"
                        />
                        {/* DEBUGGER: Pokaże Ci dokładny adres, z którego próbuje pobrać obrazek */}
                        <div className="mt-4 p-2 bg-black text-white text-[10px] font-mono break-all max-w-full text-center opacity-70">
                            URL: {imageUrl}
                        </div>
                      </>
                  ) : (
                      <div className="text-slate-300 font-bold">Image Placeholder</div>
                  )}
              </div>
          ) : null}

          <div className={`flex flex-col justify-center p-12 ${showImage ? '' : 'max-w-4xl mx-auto w-full items-center text-center'}`}>
              <div className="mb-8 w-full">
                  <h1 className="text-4xl font-black text-slate-800 mb-4 leading-tight">{formatText(data.title)}</h1>
                  <p className="text-xl text-slate-500 font-medium">{formatText(data.content)}</p>
              </div>

              <div className="w-full space-y-8">
                  {/* WORD CARD */}
                  {data.visual_type === 'word_card' && (
                    <div className={`p-8 rounded-[2rem] border-4 text-center transition-colors duration-500 w-full ${getArticleColor(data.article || '')}`}>
                      <div className="text-sm font-black uppercase opacity-60 mb-2 tracking-widest">German Word</div>
                      <div className="text-6xl font-black mb-2 tracking-tight break-words">
                        <span className="opacity-60 mr-4 text-4xl align-middle">{data.article}</span>
                        {data.german_word}
                      </div>
                      {data.plural && (
                         <div className="text-xl font-bold text-slate-600 mb-4 bg-white/50 inline-block px-4 py-1 rounded-lg">Plural: {data.plural}</div>
                      )}
                      <div className="text-2xl italic opacity-90 font-serif border-t border-black/10 pt-4 mt-2">"{formatText(data.example_sentence)}"</div>
                    </div>
                  )}

                  {/* MNEMONICS */}
                  {data.mnemonics && (
                    <div className="bg-amber-50 border-l-8 border-amber-400 p-6 rounded-r-2xl shadow-sm text-left w-full">
                      <p className="text-amber-800 font-bold text-lg leading-relaxed italic">
                        💡 {formatText(data.mnemonics)}
                      </p>
                    </div>
                  )}

                  {/* TEXT CONTENT */}
                  {(['story', 'intro', 'summary', 'fun_fact', 'dialogue'].includes(data.visual_type)) && (
                    <div className="bg-slate-50 p-8 rounded-[2rem] border-2 border-slate-100 font-serif text-2xl text-slate-700 leading-loose w-full whitespace-pre-wrap">
                      {formatText(data.example_sentence || data.content)}
                    </div>
                  )}

                  {/* FILL GAP */}
                  {data.interaction_type === 'fill_gap' && (
                      <div className="bg-indigo-50 p-8 rounded-[2rem] border-2 border-indigo-100 w-full">
                          {renderContextWithGap()}
                          {!status || status === 'idle' ? (
                              <div className="flex gap-3 justify-center flex-wrap mt-6">
                                  {['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'].map(c => (
                                      <button 
                                          key={c} 
                                          onClick={() => setLocalInput(prev => prev + c)} 
                                          className="w-12 h-12 bg-white border-2 border-indigo-200 rounded-xl font-black text-xl text-indigo-700 hover:bg-indigo-600 hover:text-white hover:border-indigo-600 transition-all shadow-md active:scale-95"
                                      >
                                          {c}
                                      </button>
                                  ))}
                              </div>
                          ) : null}
                      </div>
                  )}

                  {/* CHOICE */}
                  {data.interaction_type === 'choice' && data.options && (
                      <div className="bg-slate-50 p-8 rounded-[2rem] border-2 border-slate-100 w-full">
                           <p className="text-3xl font-bold text-slate-700 mb-8 text-center">{formatText(data.question_context)}</p>
                           <div className="flex justify-center gap-4">
                              {data.options.map((opt: string) => (
                                  <button
                                      key={opt}
                                      onClick={() => setSelectedOption(opt)}
                                      disabled={status === 'checked'}
                                      className={`px-8 py-5 rounded-2xl font-black text-2xl border-4 transition-all uppercase flex-1 ${
                                          selectedOption === opt 
                                          ? 'bg-indigo-600 text-white border-indigo-700 scale-105 shadow-xl' 
                                          : 'bg-white text-slate-600 border-slate-200 hover:border-indigo-400 hover:bg-indigo-50'
                                      } ${status === 'checked' && (data.german_word?.toLowerCase().includes(opt.toLowerCase())) ? '!bg-green-500 !border-green-600 !text-white' : ''} 
                                        ${status === 'checked' && selectedOption === opt && !isCorrect ? '!bg-red-500 !border-red-600 !text-white' : ''}
                                      `}
                                  >
                                      {opt}
                                  </button>
                              ))}
                           </div>
                      </div>
                  )}

                  {/* FEEDBACK */}
                  {status === 'checked' && (
                      <div className={`p-6 rounded-2xl text-center font-bold text-xl animate-in fade-in zoom-in duration-300 w-full ${isCorrect ? 'bg-green-100 text-green-700 border-2 border-green-200' : 'bg-red-100 text-red-700 border-2 border-red-200'}`}>
                          {isCorrect ? (
                              <div className="flex items-center justify-center gap-3">
                                  <Icon.CheckCircle size={32} /> Correct! Well done.
                              </div>
                          ) : (
                              <div className="flex flex-col items-center gap-2">
                                  <div className="flex items-center gap-2"><Icon.XCircle size={32} /> Incorrect.</div>
                                  <div className="text-slate-800 font-normal text-lg">
                                       The correct answer is: <span className="font-black bg-white px-3 py-1 rounded-lg border border-slate-200 shadow-sm">{data.german_word}</span>
                                  </div>
                              </div>
                          )}
                      </div>
                  )}

                  {/* NEXT BUTTON */}
                  <button 
                    onClick={() => {
                        if (data.interaction_type !== 'read_only' && status === 'idle') {
                            handleCheck();
                        } else {
                            onNext();
                        }
                    }}
                    className={`w-full py-6 rounded-[1.5rem] font-black text-2xl transition-all flex items-center justify-center gap-4 shadow-xl active:scale-[0.98] group mt-6 ${
                        status === 'idle' && data.interaction_type !== 'read_only'
                        ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                        : 'bg-slate-800 text-white hover:bg-slate-900'
                    }`}
                  >
                    {status === 'idle' && data.interaction_type !== 'read_only' ? 'CHECK ANSWER' : 'CONTINUE'} 
                    <Icon.ArrowRight size={32} className="group-hover:translate-x-2 transition-transform"/>
                  </button>

              </div>
          </div>
      </div>
    </div>
  );
};

// --- MAIN COMPONENT ---
const ImageLabeling: React.FC = () => {
  const [session, setSession] = useState<any>(null);
  const [currentTrial, setCurrentTrial] = useState<any>(null);
  const [userInput, setUserInput] = useState('');
  const [selectedArticle, setSelectedArticle] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<any>(null);
  
  const [showDemographics, setShowDemographics] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/experiment/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: `user_${Date.now()}`, condition }),
      });
      const data = await resp.json();
      setSession(data);
      fetchNextTrial(data.session_id);
    } catch { setError("Connection failed. Check backend."); }
    finally { setIsLoading(false); }
  };

  const fetchNextTrial = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setFeedback(null);
    setUserInput('');
    setSelectedArticle(null);
    
    try {
      const resp = await fetch(`${API_BASE}/experiment/trial/${sessionId}`);
      const data = await resp.json();
      
      if (data.status === "completed") {
        setShowDemographics(true);
      } else if (data.status === "transition") {
        fetchNextTrial(sessionId);
      } else {
        setCurrentTrial(data);
      }
    } catch { setError("Failed to load task."); }
    finally { setIsLoading(false); }
  }, []);

  const submitAnswer = async (answer?: string) => {
    if (answer === 'next_step') {
        setIsLoading(true);
        try {
            const resp = await fetch(`${API_BASE}/experiment/submit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: session.session_id, user_answer: "NEXT", start_time: 0 }),
            });
            const data = await resp.json();
            if (data.move_next) fetchNextTrial(session.session_id);
        } catch { setError("Nav error"); }
        finally { setIsLoading(false); }
        return;
    }

    let finalAnswer = answer || userInput;
    if (currentTrial?.task_type === 'type_word' && !answer) {
        if (!selectedArticle) { setError("Please select an article."); return; }
        if (!userInput.trim()) return;
        finalAnswer = `${selectedArticle} ${userInput.trim()}`;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const resp = await fetch(`${API_BASE}/experiment/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: session.session_id, user_answer: finalAnswer, start_time: 0 }),
      });
      const data = await resp.json();
      setFeedback(data);
      
      if (data.transition) {
          setTimeout(() => fetchNextTrial(session.session_id), 1500);
      } 
    } catch { setError("Submission error."); }
    finally { setIsLoading(false); }
  };

  const handleDemographicsSubmit = async (formData: any) => {
      setIsLoading(true);
      try {
          await fetch(`${API_BASE}/experiment/finalize`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  session_id: session.session_id,
                  ...formData
              })
          });
          setShowDemographics(false);
          setIsComplete(true);
      } catch {
          setError("Failed to save data.");
      } finally {
          setIsLoading(false);
      }
  };

  const skipToPhase = async (phase: string) => {
    if (!session) return;
    setIsLoading(true);
    try {
        await fetch(`${API_BASE}/experiment/skip`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: session.session_id, phase})
        });
        fetchNextTrial(session.session_id);
    } catch { setError("Skip failed"); }
    finally { setIsLoading(false); }
  };

  const formatPhase = (p: string) => {
    switch(p) {
        case 'learning': return 'Learning Phase';
        case 'pre-test': return 'Pre-Test';
        case 'post-test': return 'Post-Test';
        default: return p ? p.charAt(0).toUpperCase() + p.slice(1) : '';
    }
  };

  if (isComplete) return (
    <div className="w-full max-w-xl bg-white rounded-[2.5rem] shadow-2xl text-center p-12 border-8 border-white mx-auto mt-12">
      <Icon.CheckCircle size={64} className="text-green-500 mx-auto mb-6" />
      <h1 className="text-4xl font-black text-slate-700 mb-4 tracking-tight">Experiment Completed!</h1>
      <p className="text-slate-500 mb-8">Thank you for your participation. Your data has been saved.</p>
      <button onClick={() => window.location.reload()} className="px-10 py-4 bg-blue-600 text-white rounded-2xl font-bold shadow-lg hover:bg-blue-700 transition-all active:scale-95">Start New Session</button>
    </div>
  );

  if (showDemographics) return (<DemographicsForm onSubmit={handleDemographicsSubmit} />);

  if (!session) return (
    <div className="w-full max-w-3xl text-center">
      <h1 className="text-5xl font-black mb-12 text-slate-800 tracking-tight">Vocabulary AI Study</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 px-4">
        {['A', 'B'].map(cond => (
          <button key={cond} onClick={() => startExperiment(cond as any)} className="p-10 bg-white border-4 border-transparent hover:border-blue-400 rounded-[3rem] shadow-xl text-left flex items-start gap-6 group transition-all">
            <div className={`p-5 rounded-3xl transition-colors ${cond === 'A' ? 'bg-blue-50 text-blue-400 group-hover:bg-blue-600 group-hover:text-white' : 'bg-purple-50 text-purple-400 group-hover:bg-purple-600 group-hover:text-white'}`}>
                {cond === 'A' ? <Icon.XCircle size={40} /> : <Icon.CheckCircle size={40} />}
            </div>
            <div>
                <h2 className="text-2xl font-bold text-slate-700">Group {cond}</h2>
                <p className="text-slate-500">{cond === 'A' ? 'Static Learning' : 'Adaptive AI Learning'}</p>
            </div>
          </button>
        ))}
      </div>
      {error && <div className="mt-8 text-red-500 font-bold bg-red-100 p-4 rounded-xl">{error}</div>}
    </div>
  );

  if (currentTrial?.task_type === 'learning_step') {
      return (
        <div className="w-full flex justify-center py-8 px-4">
          <LearningScreenRenderer 
            data={currentTrial.payload} 
            onNext={() => submitAnswer('next_step')} 
          />
        </div>
      );
  }

  // --- RENDER FOR PRE/POST TEST (UPDATED) ---
  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-7xl px-4 py-8">
      <div className="w-full bg-white rounded-[2.5rem] shadow-2xl overflow-hidden border-8 border-white relative min-h-[600px] flex flex-col">
        
        {/* Header */}
        <div className="bg-slate-900 p-4 text-white flex justify-between items-center px-8 shrink-0">
          <div className="flex items-center gap-3">
              <div className="w-2.5 h-2.5 rounded-full bg-blue-400 animate-pulse shadow-[0_0_8px_rgba(96,165,250,0.6)]"></div>
              <span className="uppercase tracking-widest text-[10px] font-black opacity-80">{formatPhase(currentTrial?.phase)}</span>
          </div>
          <div className="flex items-center gap-4">
              <span className="text-[10px] font-mono opacity-60">Item {currentTrial ? currentTrial.index + 1 : 0} of {currentTrial?.total_in_phase}</span>
          </div>
        </div>

        {/* Content - GRID LAYOUT 50/50 */}
        <div className="flex-1 grid grid-cols-2">
          
          {/* Left: Image */}
          <div className="bg-slate-100 flex items-center justify-center p-10 border-r border-slate-200">
              {currentTrial && <img src={currentTrial.image_url} alt="Task Image" className="max-h-[450px] w-full object-contain transition-transform group-hover:scale-105 duration-700" />}
              {isLoading && !feedback && <div className="absolute inset-0 bg-white/50 flex items-center justify-center z-10 font-bold text-blue-500 backdrop-blur-sm">Loading...</div>}
          </div>

          {/* Right: Inputs */}
          <div className="flex flex-col justify-center p-12 bg-white">
              <div className="space-y-2 mb-8 text-left">
                <h2 className="text-4xl font-black text-slate-800 leading-tight">
                    {currentTrial?.task_type === 'article_mcq' ? 'Which article fits?' : 
                     currentTrial?.task_type === 'plural_mcq' ? 'Select the Plural Form:' : 
                     'Type the German Word:'}
                </h2>
                <p className="text-2xl text-slate-500 italic font-medium">"{currentTrial?.english_gloss}"</p>
              </div>

              {!feedback?.move_next && (
                <div className="w-full mt-2">
                    {currentTrial?.task_type !== 'type_word' ? (
                        <div className="grid grid-cols-1 gap-4">
                          {currentTrial?.options?.map((opt: string) => (
                            <button key={opt} onClick={() => submitAnswer(opt)} className="py-6 px-8 border-2 border-slate-200 rounded-2xl font-black text-xl bg-white hover:bg-blue-50 text-slate-800 transition-all text-left flex justify-between items-center group shadow-sm hover:border-blue-400">
                              {opt}
                              <Icon.ArrowRight size={24} className="opacity-0 group-hover:opacity-100 transition-all text-blue-500" />
                            </button>
                          ))}
                        </div>
                    ) : (
                        <div className="flex flex-col gap-6">
                            <div className="flex gap-4">
                              {['der', 'die', 'das'].map(art => (
                                <button key={art} onClick={() => setSelectedArticle(art)} className={`flex-1 py-5 rounded-2xl font-black text-lg uppercase tracking-widest border-2 transition-all ${selectedArticle === art ? 'bg-blue-600 text-white border-blue-700 shadow-lg' : 'bg-white text-slate-600 border-slate-300 hover:border-blue-300 hover:bg-slate-50'}`}>{art}</button>
                              ))}
                            </div>
                            <div className="flex gap-2">
                              <input autoFocus type="text" value={userInput} onChange={(e) => setUserInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && submitAnswer()} className="flex-1 p-5 border-2 border-slate-300 rounded-2xl outline-none text-2xl font-bold shadow-inner focus:border-blue-500 transition-all text-slate-900 bg-white" placeholder="Type here..." />
                              <button onClick={() => submitAnswer()} disabled={!userInput.trim() || !selectedArticle} className="px-10 py-2 bg-blue-600 text-white rounded-2xl font-black text-xl active:scale-95 shadow-lg hover:bg-blue-700 transition-all disabled:bg-slate-100 disabled:text-slate-400">SEND</button>
                            </div>
                            <div className="flex gap-3 flex-wrap">
                                {['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'].map(c => (
                                    <button key={c} onClick={()=>setUserInput(p=>p+c)} className="w-12 h-12 bg-white border-2 border-slate-200 rounded-xl font-black text-xl text-slate-600 hover:bg-blue-50 hover:border-blue-300 transition-all shadow-sm active:scale-95">{c}</button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
              )}

              {/* Feedback */}
              {feedback && (
                  <div className={`mt-6 p-8 rounded-2xl border-4 transition-all duration-500 ${feedback.is_correct ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'} shadow-lg text-left`}>
                      <div className="flex items-center gap-4 mb-4">
                          <div className={`p-3 rounded-2xl ${feedback.is_correct ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>
                            {feedback.is_correct ? <Icon.CheckCircle size={32} /> : <Icon.XCircle size={32} />}
                          </div>
                          <p className="text-2xl font-black text-slate-700 uppercase tracking-widest">{feedback.is_correct ? 'Excellent!' : 'Incorrect'}</p>
                      </div>
                      
                      <div className="space-y-4">
                          <p className="text-slate-700 text-2xl font-medium leading-relaxed">
                             {feedback.is_correct ? feedback.feedback : <span>The correct answer is: <span className="underline font-black text-red-700">{feedback.feedback?.replace("Correct: ", "")}</span></span>}
                          </p>
                      </div>
                      
                      {feedback.move_next && <button onClick={() => fetchNextTrial(session.session_id)} className="w-full py-6 mt-6 bg-slate-900 text-white rounded-[2rem] font-black text-2xl hover:bg-slate-800 transition-all flex items-center justify-center gap-4 group active:scale-[0.98] shadow-2xl">NEXT TASK <Icon.ArrowRight size={32} className="group-hover:translate-x-2 transition-transform" /></button>}
                  </div>
              )}
          </div>
        </div>
      </div>

      <div className="w-full bg-slate-800 p-4 rounded-3xl flex flex-wrap items-center justify-center gap-4 border-4 border-slate-700 shadow-xl mt-4">
        <span className="text-slate-400 font-bold text-xs uppercase tracking-widest mr-2 flex items-center gap-2">
          <Icon.Info size={14} className="text-blue-400" /> Testing Toolbar:
        </span>
        <button onClick={() => skipToPhase('pre-test')} className="px-4 py-2 bg-slate-700 text-slate-200 rounded-xl text-[10px] font-black hover:bg-blue-600 hover:text-white transition-all uppercase tracking-widest">Skip to Pre-Test</button>
        <button onClick={() => skipToPhase('learning')} className="px-4 py-2 bg-slate-700 text-slate-200 rounded-xl text-[10px] font-black hover:bg-blue-600 hover:text-white transition-all uppercase tracking-widest">Skip to Learning</button>
        <button onClick={() => skipToPhase('post-test')} className="px-4 py-2 bg-slate-700 text-slate-200 rounded-xl text-[10px] font-black hover:bg-blue-600 hover:text-white transition-all uppercase tracking-widest">Skip to Post-Test</button>
      </div>

      {error && <div className="fixed top-4 left-1/2 -translate-x-1/2 max-w-sm px-6 py-4 bg-red-600 text-white text-xs font-black rounded-full shadow-2xl flex items-center justify-center gap-2 z-[100] animate-bounce"><Icon.Info size={20} /> {error}</div>}
    </div>
  );
};

export default ImageLabeling;