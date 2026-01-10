import { useState, useEffect } from 'react';
import { API_BASE } from '../../../config/api';
import { formatText } from '../../../utils/textUtils';
import { Icon } from '../../../components/ui/Icons';

interface LearningScreenProps {
  data: any;
  onNext: () => void;
  language: 'de' | 'en';
  showLanguageSwitcher?: boolean;
  onLanguageChange?: () => void;
}

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

export const LearningScreen = ({ data, onNext, language, showLanguageSwitcher, onLanguageChange }: LearningScreenProps) => {
  const [localInput, setLocalInput] = useState("");
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'checked'>('idle');
  const [isCorrect, setIsCorrect] = useState(false);

  const t = UI_TEXTS[language];

  // (Funkcja translateBackendContent bez zmian - skrócona dla czytelności tutaj)
  const translateBackendContent = (text: string) => {
    if (!text) return text;
    if (language === 'en') return text;
    if (text.includes("First, study all")) return "Lerne zuerst diese 5 Wörter sorgfältig.";
    if (text.includes("Memorize the word")) return "Merke dir Wort, Artikel und Plural.";
    if (text.includes("Type the word (Case Sensitive!)")) return "Schreibe das Wort (Groß-/Kleinschreibung!).";
    if (text.includes("Select the correct article")) return "Wähle den richtigen Artikel:";
    if (text.includes("Starting the final test now")) return "Der Abschlusstest beginnt jetzt.";
    if (text.includes("Module Start")) return "Modul Start";
    if (text.includes("Ready!")) return "Bereit!";
    if (text.startsWith("Learn:")) return text.replace("Learn:", "Lernen:");
    if (text.startsWith("Practice:")) return text.replace("Practice:", "Üben:");
    if (text.startsWith("Gender Check:")) return text.replace("Gender Check:", "Artikel-Check:");
    if (text.startsWith("AI Plan")) return text.replace("AI Plan", "KI-Plan");
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
    let translatedContext = translateBackendContent(context);

    if (!translatedContext.includes('_______')) translatedContext += " _______";
    const parts = translatedContext.split('_______');

    return (
      <div className="flex flex-wrap items-center justify-center gap-2 text-xl font-mono bg-white p-3 rounded-xl shadow-sm border border-slate-200 leading-relaxed text-slate-800">
        <span>{formatText(parts[0])}</span>
        <input
          type="text"
          value={localInput}
          onChange={(e) => setLocalInput(e.target.value)}
          className="w-36 px-2 py-1 text-center border-b-4 border-indigo-300 bg-indigo-50 outline-none font-bold text-indigo-900 placeholder-indigo-300 focus:border-indigo-600 focus:bg-white transition-all rounded-t-lg"
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
    // GLÓWNY KONTENER:
    // Zmieniono min-h na sztywne h-[600px] (lub h-[580px] jeśli wolisz)
    // Usunięto zbędne marginesy, aby pasował do layoutu obok.
    <div className="w-full min-w-[500px] h-[600px] bg-white rounded-[2rem] shadow-xl overflow-hidden border-4 border-white flex flex-col relative">

      {/* HEADER - niski i zwarty */}
      <div className="bg-indigo-50 px-6 py-3 flex justify-between items-center border-b border-indigo-100 shrink-0 h-[60px]">
        <span className="bg-white text-indigo-700 px-3 py-1 rounded-full text-xs font-black uppercase tracking-widest shadow-sm">
          {t.step} {data.step_number}
        </span>
        <div className="flex items-center gap-3">
          {/* LANGUAGE SWITCHER FOR STATIC GROUP (Condition A) */}
          {showLanguageSwitcher && onLanguageChange && (
            <button
              onClick={onLanguageChange}
              className="flex items-center gap-2 bg-white hover:bg-indigo-100 px-3 py-1.5 rounded-full transition-colors border border-indigo-200 shadow-sm"
              title="Switch Language"
            >
              <span className={`text-xs font-bold ${language === 'en' ? 'text-indigo-700' : 'text-slate-400'}`}>EN</span>
              <div className="w-8 h-4 bg-indigo-200 rounded-full relative">
                <div className={`absolute top-0.5 w-3 h-3 bg-indigo-600 rounded-full transition-all ${language === 'de' ? 'left-[18px]' : 'left-0.5'}`}></div>
              </div>
              <span className={`text-xs font-bold ${language === 'de' ? 'text-indigo-700' : 'text-slate-400'}`}>DE</span>
            </button>
          )}
          {data.mnemonics && (
            <span className="flex items-center gap-2 text-amber-600 bg-amber-100 px-3 py-1 rounded-full text-[10px] font-black uppercase border border-amber-200">
              <Icon.Lightbulb size={14} /> {t.ai_tip}
            </span>
          )}
        </div>
      </div>

      {/* BODY - używa h-full, aby wypełnić resztę z tych 600px */}
      <div className={`flex-1 overflow-hidden ${showImage ? 'grid grid-cols-[45%_55%]' : 'flex flex-col'}`}>

        {/* LEWA KOLUMNA (OBRAZEK) */}
        {showImage && (
          <div className="bg-slate-100 h-full flex items-center justify-center p-4 border-r border-slate-200 relative">
            {imageUrl ? (
              <img
                src={imageUrl}
                alt="visual"
                onError={(e) => { e.currentTarget.style.display = 'none'; }}
                // Object-contain + max-h-full sprawia, że obrazek zawsze się zmieści w pionie
                className="w-full h-full max-h-[400px] object-contain drop-shadow-lg rounded-xl"
              />
            ) : (
              <div className="text-slate-300 font-bold">Image</div>
            )}
          </div>
        )}

        {/* PRAWA KOLUMNA (TREŚĆ) - Centrowana w pionie */}
        <div className={`flex flex-col justify-center h-full px-8 py-4 ${showImage ? '' : 'max-w-3xl mx-auto w-full items-center text-center'}`}>

          {/* TYTUŁ I OPIS */}
          <div className={`mb-4 w-full shrink-0 ${data.visual_type === 'intro' && !data.example_sentence ? 'flex-1 flex flex-col justify-center text-center' : ''}`}>
            <h1 className="text-2xl font-black text-slate-800 mb-1 leading-tight">
              {formatText(data[`title_${language}`] || translateBackendContent(data.title))}
            </h1>
            {/* Don't show content in header for text-heavy types, show in body instead */}
            {!['story', 'dialogue', 'fun_fact', 'summary'].includes(data.visual_type) && (
              <p className="text-base text-slate-500 font-medium leading-snug">
                {formatText(data[`content_${language}`] || translateBackendContent(data.content))}
              </p>
            )}
          </div>

          {/* GŁÓWNA ZAWARTOŚĆ (Word card / Input) */}
          <div className={`w-full space-y-4 flex flex-col justify-center ${['intro', 'story', 'dialogue', 'fun_fact', 'summary'].includes(data.visual_type) ? '' : 'flex-1'}`}>

            {/* WORD CARD */}
            {data.visual_type === 'word_card' && (
              <div className={`p-4 rounded-xl border-4 text-center w-full ${getArticleColor(data.article || '')}`}>
                <div className="text-[10px] font-black uppercase opacity-60 mb-1 tracking-widest">{t.german_word}</div>
                <div className="text-4xl font-black mb-1 tracking-tight">
                  <span className="opacity-60 mr-2 text-2xl align-middle">{data.article}</span>
                  {data.german_word}
                </div>
                {data.plural && (
                  <div className="text-sm font-bold text-slate-600 bg-white/50 inline-block px-2 py-0.5 rounded-md">{t.plural} {data.plural}</div>
                )}
                <div className="text-base italic opacity-90 font-serif border-t border-black/10 pt-2 mt-1">"{formatText(data.example_sentence)}"</div>
              </div>
            )}

            {/* FILL GAP */}
            {data.interaction_type === 'fill_gap' && (
              <div className="bg-indigo-50 p-5 rounded-xl border-2 border-indigo-100 w-full shadow-inner">
                {renderContextWithGap()}
                {!status || status === 'idle' ? (
                  <div className="flex gap-2 justify-center flex-wrap mt-3">
                    {['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'].map(c => (
                      <button
                        key={c}
                        onClick={() => setLocalInput(prev => prev + c)}
                        className="w-8 h-8 bg-white border border-indigo-200 rounded-lg font-bold text-base text-indigo-700 hover:bg-indigo-600 hover:text-white shadow-sm active:scale-95"
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
              <div className="bg-slate-50 p-4 rounded-xl border-2 border-slate-100 w-full">
                <p className="text-3xl font-black text-slate-800 mb-6 text-center tracking-tight">{formatText(translateBackendContent(data.question_context))}</p>
                <div className="grid grid-cols-2 gap-2">
                  {data.options.map((opt: string) => (
                    <button
                      key={opt}
                      onClick={() => setSelectedOption(opt)}
                      disabled={status === 'checked'}
                      className={`py-2 px-2 rounded-lg font-bold text-lg border-2 transition-all uppercase ${selectedOption === opt
                        ? 'bg-indigo-600 text-white border-indigo-700'
                        : 'bg-white text-slate-600 border-slate-200 hover:bg-indigo-50'
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

            {/* TEXT ONLY (Intro/Story/etc) */}
            {(['story', 'intro', 'summary', 'fun_fact', 'dialogue'].includes(data.visual_type)) && (
              <div className="p-4 rounded-xl font-serif text-lg text-slate-700 leading-relaxed text-center">
                {/* For intro, content is usually in header, but for others we put it here */}
                {/* BUT, if I hid it in header for these types, I MUST show it here. */}
                {/* Also need to handle localized content fields */}
                {formatText(data[`content_${language}`] || translateBackendContent(data.example_sentence || data.content))}
              </div>
            )}
          </div>

          {/* FOOTER AREA (Feedback + Button) */}
          <div className="mt-4 shrink-0">
            {/* FEEDBACK */}
            {status === 'checked' && (
              <div className={`mb-3 p-2 rounded-lg text-center font-bold text-sm animate-in fade-in zoom-in duration-200 ${isCorrect ? 'bg-green-100 text-green-700 border border-green-200' : 'bg-red-100 text-red-700 border border-red-200'}`}>
                {isCorrect ? (
                  <span className="flex items-center justify-center gap-2"><Icon.CheckCircle size={18} /> {t.correct}</span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <Icon.XCircle size={18} /> {t.incorrect}
                    <span className="font-normal text-slate-600">({data.german_word})</span>
                  </span>
                )}
              </div>
            )}

            <button
              onClick={() => {
                if (data.interaction_type !== 'read_only' && status === 'idle') {
                  handleCheck();
                } else {
                  onNext();
                }
              }}
              className={`w-full py-3 rounded-xl font-black text-lg transition-all flex items-center justify-center gap-2 shadow-md active:scale-[0.98] group ${status === 'idle' && data.interaction_type !== 'read_only'
                ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                : 'bg-slate-800 text-white hover:bg-slate-900'
                }`}
            >
              {status === 'idle' && data.interaction_type !== 'read_only' ? t.check_answer : t.continue}
              <Icon.ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};