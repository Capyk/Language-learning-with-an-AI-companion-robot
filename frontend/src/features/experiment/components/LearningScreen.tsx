import { useState, useEffect } from 'react';
import { API_BASE } from '../../../config/api';
import { formatText } from '../../../utils/textUtils';
import { Icon } from '../../../components/ui/Icons';

interface LearningScreenProps {
  data: any;
  onNext: () => void;
  language: 'de' | 'en'; // <--- NOWY PROP
}

// Słownik tłumaczeń dla elementów UI
const UI_TEXTS = {
    de: {
        step: "Schritt",
        ai_tip: "KI-Tipp",
        german_word: "Deutsches Wort",
        plural: "Plural:",
        check_answer: "ANTWORT PRÜFEN",
        continue: "WEITER",
        correct: "Richtig! Gut gemacht.",
        incorrect: "Falsch.",
        correct_answer_is: "Die richtige Antwort ist:",
        placeholder: "Tippen...",
        module_start: "Modul Start",
        ready: "Bereit!"
    },
    en: {
        step: "Step",
        ai_tip: "AI Tip",
        german_word: "German Word",
        plural: "Plural:",
        check_answer: "CHECK ANSWER",
        continue: "CONTINUE",
        correct: "Correct! Well done.",
        incorrect: "Incorrect.",
        correct_answer_is: "The correct answer is:",
        placeholder: "Type here...",
        module_start: "Module Start",
        ready: "Ready!"
    }
};

export const LearningScreen = ({ data, onNext, language }: LearningScreenProps) => {
  const [localInput, setLocalInput] = useState("");
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'checked'>('idle');
  const [isCorrect, setIsCorrect] = useState(false);

  // Wybór tekstów na podstawie języka
  const t = UI_TEXTS[language]; 

  // --- Funkcja tłumacząca treści z Backendu "w locie" ---
  // Backend wysyła treści po angielsku. Jeśli język to 'de', musimy je przetłumaczyć.
  const translateBackendContent = (text: string) => {
      if (!text) return text;
      if (language === 'en') return text; // Jeśli angielski, zwracamy oryginał

      // Mapowanie fraz z backendu na niemiecki
      if (text.includes("First, study all")) return "Lerne zuerst diese 5 Wörter sorgfältig.";
      if (text.includes("Memorize the word")) return "Merke dir Wort, Artikel und Plural.";
      if (text.includes("Type the word (Case Sensitive!)")) return "Schreibe das Wort (Groß-/Kleinschreibung!).";
      if (text.includes("Select the correct article")) return "Wähle den richtigen Artikel:";
      if (text.includes("Starting the final test now")) return "Der Abschlusstest beginnt jetzt.";
      if (text.includes("Module Start")) return "Modul Start";
      if (text.includes("Ready!")) return "Bereit!";
      
      // Tłumaczenie nagłówków (np. "Learn: Dog")
      if (text.startsWith("Learn:")) return text.replace("Learn:", "Lernen:");
      if (text.startsWith("Practice:")) return text.replace("Practice:", "Üben:");
      if (text.startsWith("Gender Check:")) return text.replace("Gender Check:", "Artikel-Check:");
      if (text.startsWith("AI Plan")) return text.replace("AI Plan", "KI-Plan").replace("Focusing on", "Fokus auf").replace("items", "Elemente");
      
      return text;
  };

  useEffect(() => {
    setLocalInput(""); setSelectedOption(null); setStatus('idle'); setIsCorrect(false);
  }, [data.step_number]);

  const getArticleColor = (art: string) => {
    if (!art) return 'text-slate-800 bg-slate-50';
    const lower = art.toLowerCase();
    if (lower.includes('der')) return 'text-blue-600 bg-blue-50 border-blue-200';
    if (lower.includes('die')) return 'text-red-600 bg-red-50 border-red-200';
    if (lower.includes('das')) return 'text-green-600 bg-green-50 border-green-200';
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
    // Tłumaczenie kontekstu (jeśli to instrukcja)
    let translatedContext = translateBackendContent(context);
    
    if (!translatedContext.includes('_______')) translatedContext += " _______";
    const parts = translatedContext.split('_______');
    
    return (
        <div className="flex flex-wrap items-center justify-center gap-2 text-2xl font-mono bg-white p-6 rounded-2xl shadow-sm border border-slate-200 leading-relaxed text-slate-800">
            <span>{formatText(parts[0])}</span>
            <input 
                type="text" 
                value={localInput} 
                onChange={(e) => setLocalInput(e.target.value)} 
                className="w-48 px-2 py-1 text-center border-b-4 border-indigo-300 bg-indigo-50 outline-none font-bold text-indigo-900 placeholder-indigo-300 focus:border-indigo-600 focus:bg-white transition-all rounded-t-lg" 
                disabled={status === 'checked'} 
                autoFocus 
            />
            <span>{formatText(parts[1])}</span>
        </div>
    );
  };

  const isFullWidth = ['story', 'intro', 'summary', 'fun_fact', 'dialogue'].includes(data.visual_type);
  const showImage = !isFullWidth && !!data.image_url;
  const imageUrl = data.image_url ? (data.image_url.startsWith('http') ? data.image_url : `${API_BASE}${data.image_url}`) : null;

  return (
    <div className="w-full max-w-7xl bg-white rounded-[2.5rem] shadow-2xl overflow-hidden border-8 border-white mx-auto animate-in fade-in slide-in-from-bottom-8 duration-700 flex flex-col min-h-[600px]">
      
      <div className="bg-indigo-50 p-6 flex justify-between items-center border-b border-indigo-100 shrink-0">
        <span className="bg-white text-indigo-700 px-4 py-2 rounded-full text-sm font-black uppercase tracking-widest shadow-sm">
          {t.step} {data.step_number}
        </span>
        {data.mnemonics && (
          <span className="flex items-center gap-2 text-amber-600 bg-amber-100 px-4 py-2 rounded-full text-xs font-black uppercase border border-amber-200">
            <Icon.Lightbulb size={16} /> {t.ai_tip}
          </span>
        )}
      </div>

      <div className={`flex-1 ${showImage ? 'grid grid-cols-2' : 'flex flex-col'}`}>
          {showImage && (
              <div className="bg-slate-100 flex flex-col items-center justify-center p-8 border-r border-slate-200 h-full relative overflow-hidden">
                  {imageUrl ? (
                      <img 
                          src={imageUrl} 
                          alt="visual" 
                          onError={(e) => { e.currentTarget.style.display='none'; }}
                          className="w-auto h-auto max-w-full max-h-[450px] object-contain drop-shadow-2xl rounded-2xl transition-transform hover:scale-105 duration-500"
                      />
                  ) : (
                      <div className="text-slate-300 font-bold">Image Placeholder</div>
                  )}
              </div>
          )}

          <div className={`flex flex-col justify-center p-12 ${showImage ? '' : 'max-w-4xl mx-auto w-full items-center text-center'}`}>
              <div className="mb-8 w-full">
                  {/* --- TYTUŁ I OPIS (PRZETŁUMACZONE) --- */}
                  <h1 className="text-4xl font-black text-slate-800 mb-4 leading-tight">{formatText(translateBackendContent(data.title))}</h1>
                  <p className="text-xl text-slate-500 font-medium">{formatText(translateBackendContent(data.content))}</p>
              </div>

              <div className="w-full space-y-8">
                  {/* WORD CARD */}
                  {data.visual_type === 'word_card' && (
                    <div className={`p-8 rounded-[2rem] border-4 text-center transition-colors duration-500 w-full ${getArticleColor(data.article || '')}`}>
                      <div className="text-sm font-black uppercase opacity-60 mb-2 tracking-widest">{t.german_word}</div>
                      <div className="text-6xl font-black mb-2 tracking-tight break-words">
                        <span className="opacity-60 mr-4 text-4xl align-middle">{data.article}</span>
                        {data.german_word}
                      </div>
                      {data.plural && (
                         <div className="text-xl font-bold text-slate-600 mb-4 bg-white/50 inline-block px-4 py-1 rounded-lg">{t.plural} {data.plural}</div>
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
                      {formatText(translateBackendContent(data.example_sentence || data.content))}
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
                            <p className="text-3xl font-bold text-slate-700 mb-8 text-center">{formatText(translateBackendContent(data.question_context))}</p>
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
                                  <Icon.CheckCircle size={32} /> {t.correct}
                              </div>
                          ) : (
                              <div className="flex flex-col items-center gap-2">
                                  <div className="flex items-center gap-2"><Icon.XCircle size={32} /> {t.incorrect}</div>
                                  <div className="text-slate-800 font-normal text-lg">
                                        {t.correct_answer_is} <span className="font-black bg-white px-3 py-1 rounded-lg border border-slate-200 shadow-sm">{data.german_word}</span>
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
                    {status === 'idle' && data.interaction_type !== 'read_only' ? t.check_answer : t.continue} 
                    <Icon.ArrowRight size={32} className="group-hover:translate-x-2 transition-transform"/>
                  </button>

              </div>
          </div>
      </div>
    </div>
  );
};