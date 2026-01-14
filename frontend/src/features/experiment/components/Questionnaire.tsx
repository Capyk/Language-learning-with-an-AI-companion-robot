import { useState, useEffect } from 'react';
import { Icon } from '../../../components/ui/Icons';

interface QuestionnaireProps {
    onSubmit: (data: any) => void;
    condition?: string;
    tutorUsed?: boolean;
}

export const Questionnaire = ({ onSubmit, condition, tutorUsed }: QuestionnaireProps) => {
    const [answers, setAnswers] = useState<any>({});

    // Zmiana na skalę 4-stopniową (wymusza wybór pozytywny/negatywny)
    const scale = [1, 2, 3, 4];

    const sections = [
        {
            title: "Section A – Perceived Usefulness",
            description: "Focus on the feedback you received.",
            questions: [
                "The feedback helped me understand why my answer was correct or incorrect.",
                "The feedback supported my learning of German vocabulary.",
                "The feedback helped me correct my mistakes.",
                "The feedback was appropriate for my level of German.",
                "The feedback helped me learn from my errors rather than just showing the solution."
            ]
        },
        {
            title: "Section B – Usability",
            description: "Evaluate the system interface.",
            questions: [
                "The interface was easy to understand.",
                "The tasks were clearly explained.",
                "I always knew what I was supposed to do next.",
                "The interaction felt intuitive.",
                "Overall, the system was user-friendly."
            ]
        },
        {
            title: "Section C – Engagement & Attention",
            description: "How did you feel during the session?",
            questions: [
                "I found the learning tasks engaging.",
                "The feedback kept me interested in the task.",
                "I felt motivated to continue during the learning session.",
                "I was actively thinking about my answers.",
                "Time passed quickly while using the system."
            ]
        },
        {
            title: "Section D – Overall Evaluation",
            description: "General satisfaction.",
            questions: [
                "Overall, I had a positive experience using this system.",
                "The system supported my learning effectively.",
                "I would like to use a similar system for learning vocabulary.",
                "The feedback added value to the learning experience.",
                "Overall, I am satisfied with my experience in this study."
            ]
        },
        {
            title: "Section E – AI Tutor Chat Evaluation",
            description: "Evaluate the chat assistant (if applicable).",
            questions: [
                "The AI tutor’s explanations were clear and easy to understand.",
                "The AI tutor provided helpful guidance when I was unsure.",
                "The AI tutor’s responses felt relevant to my mistakes.",
                "The AI tutor helped me reflect on my answers rather than just giving solutions.",
                "I would like to use the AI tutor chat in future learning sessions."
            ]
        }
    ];

    // FIX: Jeśli condition == 'A', usuwamy sekcję E (Tutor Eval) -> LUB jeśli condition == 'B' ale tutorUsed == false
    const activeSections = (condition === 'A' || (condition === 'B' && !tutorUsed))
        ? sections.filter(s => !s.title.includes("Section E"))
        : sections;

    const totalLikertQuestions = activeSections.reduce((acc, sec) => acc + sec.questions.length, 0);
    const filledLikert = Object.keys(answers).filter(k => k.startsWith('q_')).length;

    // Walidacja: Wszystkie pytania zamknięte muszą być wypełnione
    const countChars = (str?: string) => {
        if (!str) return 0;
        return str.trim().length;
    };

    const MIN_CHARS = 15;
    const openQuestions = ['open_like', 'open_confusing', 'open_suggestions'];

    // Check if open questions are valid (>= MIN_CHARS)
    const openValid = openQuestions.every(key => {
        const val = answers[key] || "";
        return countChars(val) >= MIN_CHARS;
    });

    const isComplete = (filledLikert >= totalLikertQuestions) && openValid;

    const handleChange = (id: string, val: any) => {
        setAnswers((prev: any) => ({ ...prev, [id]: val }));
    };

    useEffect(() => { window.scrollTo(0, 0); }, []);

    return (
        <div className="w-full max-w-6xl mx-auto my-8 animate-in fade-in slide-in-from-bottom-8 duration-700 px-4">

            {/* Header */}
            <div className="bg-white border-l-8 border-indigo-600 rounded-lg shadow-sm p-8 mb-8">
                <h1 className="text-3xl font-black text-slate-900 mb-2">Final Questionnaire</h1>
                <p className="text-slate-600 text-lg">
                    Please answer the following questions using a <strong>4-point scale</strong>. <br />
                    (1 = Strongly Disagree, 4 = Strongly Agree). There is no "Neutral" option.
                </p>
            </div>

            {/* Likert Sections A-E */}
            {activeSections.map((sec, sIdx) => (
                <div key={sIdx} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mb-8">
                    {/* Nagłówek sekcji */}
                    <div className="bg-slate-50 px-8 py-5 border-b border-slate-200">
                        <h2 className="text-2xl font-bold text-slate-800">{sec.title}</h2>
                        <p className="text-slate-500">{sec.description}</p>
                    </div>

                    {/* Lista pytań */}
                    <div className="divide-y divide-slate-100">
                        {sec.questions.map((q, qIdx) => {
                            const qId = `q_${sIdx}_${qIdx}`;
                            const isAnswered = answers[qId] !== undefined;

                            return (
                                <div key={qId} className={`flex flex-col lg:flex-row lg:items-center justify-between gap-4 px-8 py-5 hover:bg-slate-50 transition-colors ${isAnswered ? 'bg-indigo-50/30' : ''}`}>

                                    {/* LEWA STRONA: Pytanie */}
                                    <div className="lg:w-1/2">
                                        <p className="font-medium text-slate-800 text-lg leading-snug">
                                            {q} {!isAnswered && <span className="text-red-500 font-bold ml-1">*</span>}
                                        </p>
                                    </div>

                                    {/* PRAWA STRONA: Skala 1-4 */}
                                    <div className="lg:w-1/2 flex items-center justify-between lg:justify-end gap-3 sm:gap-6">
                                        <span className="text-[10px] font-bold text-slate-400 uppercase text-right w-20 leading-tight hidden sm:block">Strongly Disagree</span>

                                        <div className="flex gap-3 sm:gap-4">
                                            {scale.map(v => (
                                                <label key={v} className="group relative cursor-pointer">
                                                    <input
                                                        type="radio"
                                                        name={qId}
                                                        value={v}
                                                        onChange={() => handleChange(qId, v)}
                                                        checked={answers[qId] === v}
                                                        className="peer sr-only"
                                                    />
                                                    <div className="w-12 h-12 flex items-center justify-center rounded-full border-2 border-slate-300 bg-white text-slate-500 font-bold text-lg transition-all peer-checked:border-indigo-600 peer-checked:bg-indigo-600 peer-checked:text-white peer-checked:scale-110 group-hover:border-indigo-300 shadow-sm">
                                                        {v}
                                                    </div>
                                                </label>
                                            ))}
                                        </div>

                                        <span className="text-[10px] font-bold text-slate-400 uppercase text-left w-20 leading-tight hidden sm:block">Strongly Agree</span>
                                    </div>

                                    {/* Mobile labels */}
                                    <div className="flex justify-between w-full sm:hidden text-xs text-slate-400 font-bold mt-1 px-1">
                                        <span>Disagree</span>
                                        <span>Agree</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            ))}



            {/* Section F - Open Ended */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 mb-8 border-l-8 border-l-slate-700">
                <h2 className="text-2xl font-bold text-slate-800 mb-6">Section F – Open Ended</h2>
                <div className="bg-amber-50 text-amber-800 p-4 rounded-lg mb-6 text-sm font-bold border border-amber-200">
                    Please provide detailed answers. Minimum {MIN_CHARS} characters required per question.
                </div>

                <div className="grid gap-8">
                    {/* Q26 */}
                    <div>
                        <label className="block text-slate-700 font-bold mb-3 text-lg">26. What did you like most about the feedback?</label>
                        <textarea
                            onChange={e => handleChange('open_like', e.target.value)}
                            value={answers['open_like'] || ''}
                            rows={3}
                            className={`w-full p-4 border rounded-xl focus:ring-2 outline-none transition-all placeholder-slate-400 text-base text-slate-900 bg-white ${(answers['open_like'] && countChars(answers['open_like']) < MIN_CHARS)
                                ? 'border-red-300 focus:ring-red-500 bg-red-50'
                                : 'border-slate-300 focus:ring-indigo-500 bg-white focus:bg-white'
                                }`}
                            placeholder="Your answer..."
                        ></textarea>
                        <div className={`text-xs mt-1 font-bold ${countChars(answers['open_like']) < MIN_CHARS ? 'text-red-500' : 'text-green-600'}`}>
                            Character count: {countChars(answers['open_like'])} / {MIN_CHARS}
                        </div>
                    </div>

                    {/* Q27 */}
                    <div>
                        <label className="block text-slate-700 font-bold mb-3 text-lg">27. What did you find confusing or unhelpful?</label>
                        <textarea
                            onChange={e => handleChange('open_confusing', e.target.value)}
                            value={answers['open_confusing'] || ''}
                            rows={3}
                            className={`w-full p-4 border rounded-xl focus:ring-2 outline-none transition-all placeholder-slate-400 text-base text-slate-900 bg-white ${(answers['open_confusing'] && countChars(answers['open_confusing']) < MIN_CHARS)
                                ? 'border-red-300 focus:ring-red-500 bg-red-50'
                                : 'border-slate-300 focus:ring-indigo-500 bg-white focus:bg-white'
                                }`}
                            placeholder="Your answer..."
                        ></textarea>
                        <div className={`text-xs mt-1 font-bold ${countChars(answers['open_confusing']) < MIN_CHARS ? 'text-red-500' : 'text-green-600'}`}>
                            Character count: {countChars(answers['open_confusing'])} / {MIN_CHARS}
                        </div>
                    </div>

                    {/* Q28 */}
                    <div>
                        <label className="block text-slate-700 font-bold mb-3 text-lg">28. Do you have suggestions for improving the system?</label>
                        <textarea
                            onChange={e => handleChange('open_suggestions', e.target.value)}
                            value={answers['open_suggestions'] || ''}
                            rows={3}
                            className={`w-full p-4 border rounded-xl focus:ring-2 outline-none transition-all placeholder-slate-400 text-base text-slate-900 bg-white ${(answers['open_suggestions'] && countChars(answers['open_suggestions']) < MIN_CHARS)
                                ? 'border-red-300 focus:ring-red-500 bg-red-50'
                                : 'border-slate-300 focus:ring-indigo-500 bg-white focus:bg-white'
                                }`}
                            placeholder="Your answer..."
                        ></textarea>
                        <div className={`text-xs mt-1 font-bold ${countChars(answers['open_suggestions']) < MIN_CHARS ? 'text-red-500' : 'text-green-600'}`}>
                            Character count: {countChars(answers['open_suggestions'])} / {MIN_CHARS}
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex justify-end pb-16">
                <button
                    disabled={!isComplete}
                    onClick={() => onSubmit(answers)}
                    className="px-12 py-5 bg-slate-900 text-white font-black rounded-2xl text-xl hover:bg-black shadow-xl disabled:bg-slate-300 disabled:cursor-not-allowed disabled:shadow-none transition-all active:scale-95 flex items-center gap-3"
                >
                    NEXT STEP
                    <Icon.ArrowRight />
                </button>
            </div>
        </div >
    );
};