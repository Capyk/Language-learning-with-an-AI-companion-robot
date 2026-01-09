import { useState, useEffect } from 'react';
import { API_BASE } from '../../../config/api';
import { Icon } from '../../../components/ui/Icons';
import { FeedbackCard } from './FeedbackCard';

interface ExperimentLayoutProps {
    trial: any;
    feedback: any;
    isLoading: boolean;
    onSubmit: (answer: string) => void;
    onSkip: (phase: string) => void;
    onNextTrial: () => void; // Wywoływane z FeedbackCard
}

export const ExperimentLayout = ({ trial, feedback, isLoading, onSubmit, onSkip, onNextTrial }: ExperimentLayoutProps) => {
    const [localInput, setLocalInput] = useState('');
    const [selectedArticle, setSelectedArticle] = useState<string | null>(null);

    // Reset stanu przy zmianie zadania
    useEffect(() => {
        setLocalInput('');
        setSelectedArticle(null);
    }, [trial]);

    const handleSubmit = (option?: string) => {
        if (option) {
            onSubmit(option);
        } else {
            // Walidacja dla wpisywania ręcznego
            if (trial.task_type === 'type_word') {
                if (!selectedArticle) return alert("Please select an article!");
                if (!localInput.trim()) return;
                onSubmit(`${selectedArticle} ${localInput.trim()}`);
            } else {
                onSubmit(localInput);
            }
        }
    };

    const trialImageUrl = trial?.image_url 
        ? (trial.image_url.startsWith('http') ? trial.image_url : `${API_BASE}${trial.image_url}`)
        : null;

    const formatPhase = (p: string) => {
        switch(p) {
            case 'learning': return 'Learning Phase';
            case 'pre-test': return 'Pre-Test';
            case 'post-test': return 'Post-Test';
            default: return p ? p.charAt(0).toUpperCase() + p.slice(1) : '';
        }
    };

    return (
        <div className="w-full bg-white rounded-[2.5rem] shadow-2xl overflow-hidden border-8 border-white relative min-h-[600px] flex flex-col">
            {/* Header */}
            <div className="bg-slate-900 p-4 text-white flex justify-between items-center px-8 shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-2.5 h-2.5 rounded-full bg-blue-400 animate-pulse shadow-[0_0_8px_rgba(96,165,250,0.6)]"></div>
                    <span className="uppercase tracking-widest text-[10px] font-black opacity-80">{formatPhase(trial?.phase)}</span>
                </div>
                <div className="flex items-center gap-4">
                    <span className="text-[10px] font-mono opacity-60">Item {trial ? trial.index + 1 : 0} of {trial?.total_in_phase}</span>
                </div>
            </div>

            {/* Content Split */}
            <div className="flex-1 grid grid-cols-2">
                {/* Left: Image */}
                <div className="bg-slate-100 flex flex-col items-center justify-center p-8 border-r border-slate-200 h-full relative overflow-hidden">
                    {trialImageUrl ? (
                        <img 
                            src={trialImageUrl} 
                            alt="Task" 
                            className="w-auto h-auto max-w-full max-h-[450px] object-contain transition-transform group-hover:scale-105 duration-700" 
                            onError={(e) => { e.currentTarget.style.display='none'; }}
                        />
                    ) : (
                        <div className="text-slate-300 font-bold">Image Placeholder</div>
                    )}
                    {isLoading && !feedback && <div className="absolute inset-0 bg-white/50 flex items-center justify-center z-10 font-bold text-blue-500 backdrop-blur-sm">Loading...</div>}
                </div>

                {/* Right: Interaction */}
                <div className="flex flex-col justify-center p-12 bg-white">
                    <div className="space-y-2 mb-8 text-left">
                        <h2 className="text-4xl font-black text-slate-800 leading-tight">
                            {trial?.task_type === 'article_mcq' ? 'Which article fits?' : 
                             trial?.task_type === 'plural_mcq' ? 'Select the Plural Form:' : 
                             'Type the German Word:'}
                        </h2>
                        <p className="text-2xl text-slate-500 italic font-medium">"{trial?.english_gloss}"</p>
                    </div>

                    {!feedback ? (
                        <div className="w-full mt-2">
                            {trial?.task_type !== 'type_word' ? (
                                // MCQ Options
                                <div className="grid grid-cols-1 gap-4">
                                    {trial?.options?.map((opt: string) => (
                                        <button key={opt} onClick={() => handleSubmit(opt)} className="py-6 px-8 border-2 border-slate-200 rounded-2xl font-black text-xl bg-white hover:bg-blue-50 text-slate-800 transition-all text-left flex justify-between items-center group shadow-sm hover:border-blue-400">
                                            {opt}
                                            <Icon.ArrowRight size={24} className="opacity-0 group-hover:opacity-100 transition-all text-blue-500" />
                                        </button>
                                    ))}
                                </div>
                            ) : (
                                // Typing Interface
                                <div className="flex flex-col gap-6">
                                    <div className="flex gap-4">
                                        {['der', 'die', 'das'].map(art => (
                                            <button key={art} onClick={() => setSelectedArticle(art)} className={`flex-1 py-5 rounded-2xl font-black text-lg uppercase tracking-widest border-2 transition-all ${selectedArticle === art ? 'bg-blue-600 text-white border-blue-700 shadow-lg' : 'bg-white text-slate-600 border-slate-300 hover:border-blue-300 hover:bg-slate-50'}`}>{art}</button>
                                        ))}
                                    </div>
                                    <div className="flex gap-2">
                                        <input autoFocus type="text" value={localInput} onChange={(e) => setLocalInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSubmit()} className="flex-1 p-5 border-2 border-slate-300 rounded-2xl outline-none text-2xl font-bold shadow-inner focus:border-blue-500 transition-all text-slate-900 bg-white" placeholder="Type here..." />
                                        <button onClick={() => handleSubmit()} disabled={!localInput.trim() || !selectedArticle} className="px-10 py-2 bg-blue-600 text-white rounded-2xl font-black text-xl active:scale-95 shadow-lg hover:bg-blue-700 transition-all disabled:bg-slate-100 disabled:text-slate-400">SEND</button>
                                    </div>
                                    <div className="flex gap-3 flex-wrap">
                                        {['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'].map(c => (
                                            <button key={c} onClick={()=>setLocalInput(p=>p+c)} className="w-12 h-12 bg-white border-2 border-slate-200 rounded-xl font-black text-xl text-slate-600 hover:bg-blue-50 hover:border-blue-300 transition-all shadow-sm active:scale-95">{c}</button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <FeedbackCard feedback={feedback} onNext={onNextTrial} />
                    )}
                </div>
            </div>

            {/* Dev Tools Footer (można usunąć na produkcji) */}
            <div className="absolute bottom-2 left-2 flex gap-2 opacity-20 hover:opacity-100 transition-opacity">
                <button onClick={() => onSkip('pre-test')} className="p-1 bg-black text-white text-[10px]">Skip to Pre</button>
                <button onClick={() => onSkip('learning')} className="p-1 bg-black text-white text-[10px]">Skip to Learn</button>
                <button onClick={() => onSkip('post-test')} className="p-1 bg-black text-white text-[10px]">Skip to Post</button>
            </div>
        </div>
    );
};