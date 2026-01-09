import React, { useState, useEffect } from 'react';
import { Icon } from '../../../components/ui/Icons';

interface QuestionnaireProps {
    onSubmit: (data: any) => void;
}

export const Questionnaire = ({ onSubmit }: QuestionnaireProps) => {
    const [answers, setAnswers] = useState<any>({});
    
    const scale = [1, 2, 3, 4, 5];
    
    const sections = [
        { 
            title: "Perceived Usefulness", 
            description: "Did the system help you learn?",
            questions: [
                "Feedback helped me understand errors", "Supported my learning", "Relevant to task", 
                "Helped notice patterns", "Helped correct mistakes", "Improved understanding", 
                "Made learning effective", "Appropriate level", "Learned from errors", "Useful for future"
            ] 
        },
        { 
            title: "Usability", 
            description: "Was the system easy to use?",
            questions: [
                "Interface easy", "Tasks clear", "Layout consistent", "Knew what to do", 
                "Easy to read", "Feedback clear", "Easy without instructions", "No difficulties", 
                "Intuitive", "User-friendly"
            ] 
        },
        { 
            title: "Engagement", 
            description: "How did you feel during the tasks?",
            questions: [
                "Tasks engaging", "Feedback interesting", "Focused", "Motivated", "Held attention", 
                "Not bored", "Made tasks interesting", "Actively thinking", "Time passed quickly", 
                "Encouraged attention"
            ] 
        },
        { 
            title: "Overall Experience", 
            description: "General satisfaction.",
            questions: [
                "Positive experience", "Effective learning", "Would use again", "Suitable for beginners", 
                "Feedback added value", "Felt comfortable", "Met expectations", "Recommend to others", 
                "Made learning easier", "Satisfied"
            ] 
        }
    ];

    const totalLikertQuestions = sections.reduce((acc, sec) => acc + sec.questions.length, 0);
    const filledLikert = Object.keys(answers).filter(k => k.startsWith('q_')).length;
    const isComplete = filledLikert >= totalLikertQuestions;

    const handleChange = (id: string, val: any) => {
        setAnswers((prev: any) => ({...prev, [id]: val}));
    };

    useEffect(() => { window.scrollTo(0,0); }, []);

    return (
        <div className="w-full max-w-5xl mx-auto my-8 animate-in fade-in slide-in-from-bottom-8 duration-700 px-4">
            
            {/* Header - bardziej kompaktowy */}
            <div className="bg-white border-l-8 border-indigo-600 rounded-lg shadow-sm p-6 mb-6">
                <h1 className="text-3xl font-black text-slate-900 mb-1">Feedback Survey</h1>
                <p className="text-slate-600">Please answer honestly to help us improve.</p>
            </div>

            {sections.map((sec, sIdx) => (
                <div key={sIdx} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mb-6">
                    {/* Nagłówek sekcji */}
                    <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
                        <h2 className="text-xl font-bold text-slate-800">{sec.title}</h2>
                        <p className="text-slate-500 text-sm">{sec.description}</p>
                    </div>
                    
                    {/* Lista pytań - Compact Row Layout */}
                    <div className="divide-y divide-slate-100">
                        {sec.questions.map((q, qIdx) => {
                            const qId = `q_${sIdx}_${qIdx}`;
                            const isAnswered = answers[qId] !== undefined;

                            return (
                                <div key={qId} className={`flex flex-col md:flex-row md:items-center justify-between gap-4 px-6 py-4 hover:bg-slate-50 transition-colors ${isAnswered ? 'bg-indigo-50/30' : ''}`}>
                                    
                                    {/* LEWA STRONA: Pytanie */}
                                    <div className="md:w-5/12 lg:w-1/2">
                                        <p className="font-medium text-slate-800 text-lg leading-tight">
                                            {q} {!isAnswered && <span className="text-red-400 text-sm align-top">*</span>}
                                        </p>
                                    </div>

                                    {/* PRAWA STRONA: Skala */}
                                    <div className="md:w-7/12 lg:w-1/2 flex items-center justify-between md:justify-end gap-2 sm:gap-4">
                                        <span className="text-[10px] font-bold text-slate-400 uppercase text-right w-16 leading-tight hidden sm:block">Strongly Disagree</span>
                                        
                                        <div className="flex gap-2 sm:gap-3">
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
                                                    {/* Zmniejszyłem trochę kółka (w-10 h-10 zamiast 12) żeby było bardziej zbite */}
                                                    <div className="w-10 h-10 flex items-center justify-center rounded-full border-2 border-slate-300 bg-white text-slate-500 font-bold transition-all peer-checked:border-indigo-600 peer-checked:bg-indigo-600 peer-checked:text-white peer-checked:scale-110 group-hover:border-indigo-300 shadow-sm">
                                                        {v}
                                                    </div>
                                                </label>
                                            ))}
                                        </div>
                                        
                                        <span className="text-[10px] font-bold text-slate-400 uppercase text-left w-16 leading-tight hidden sm:block">Strongly Agree</span>
                                    </div>

                                    {/* Mobile labels (tylko na małych ekranach pod kropkami) */}
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

            {/* Additional Comments - też bardziej kompaktowe */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-6">
                <h2 className="text-xl font-bold text-slate-800 mb-4">Final Thoughts (Optional)</h2>
                <div className="grid md:grid-cols-2 gap-6">
                    <div>
                        <label className="block text-slate-700 font-bold mb-2 text-sm">What did you like most?</label>
                        <textarea 
                            onChange={e => handleChange('open_like', e.target.value)} 
                            rows={3} 
                            className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none transition-all text-slate-900 bg-white placeholder-slate-400 text-sm" 
                            placeholder="..."
                        ></textarea>
                    </div>
                    <div>
                        <label className="block text-slate-700 font-bold mb-2 text-sm">Any confusion or issues?</label>
                        <textarea 
                            onChange={e => handleChange('open_confusing', e.target.value)} 
                            rows={3} 
                            className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none transition-all text-slate-900 bg-white placeholder-slate-400 text-sm" 
                            placeholder="..."
                        ></textarea>
                    </div>
                </div>
            </div>

            <div className="flex justify-end pb-12">
                <button 
                    disabled={!isComplete} 
                    onClick={() => onSubmit(answers)} 
                    className="px-10 py-4 bg-slate-900 text-white font-black rounded-xl text-lg hover:bg-black shadow-xl disabled:bg-slate-300 disabled:cursor-not-allowed disabled:shadow-none transition-all active:scale-95 flex items-center gap-3"
                >
                    NEXT STEP
                    <Icon.ArrowRight />
                </button>
            </div>
        </div>
    );
};