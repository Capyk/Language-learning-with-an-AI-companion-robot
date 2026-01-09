import React from 'react';
import { Icon } from '../../../components/ui/Icons';

interface FeedbackCardProps {
    feedback: {
        score: number;
        feedback: string;
        move_next?: boolean;
    };
    onNext: () => void;
}

export const FeedbackCard = ({ feedback, onNext }: FeedbackCardProps) => {
    const isCorrect = feedback.score === 1.0;
    const isPartial = feedback.score === 0.5;

    return (
        <div className={`mt-6 p-8 rounded-2xl border-4 transition-all duration-500 shadow-lg text-left ${
            isCorrect ? 'bg-green-50 border-green-100' : 
            isPartial ? 'bg-yellow-50 border-yellow-100' : 
            'bg-red-50 border-red-100'
        }`}>
            <div className="flex items-center gap-4 mb-4">
                <div className={`p-3 rounded-2xl ${
                    isCorrect ? 'bg-green-100 text-green-600' : 
                    isPartial ? 'bg-yellow-100 text-yellow-600' : 
                    'bg-red-100 text-red-600'
                }`}>
                    {feedback.score > 0 ? <Icon.CheckCircle size={32} /> : <Icon.XCircle size={32} />}
                </div>
                <p className="text-2xl font-black text-slate-700 uppercase tracking-widest">
                    {isCorrect ? 'Excellent!' : isPartial ? 'Almost Correct!' : 'Incorrect'}
                </p>
            </div>

            <div className="space-y-4">
                <p className="text-slate-700 text-2xl font-medium leading-relaxed">
                    {isCorrect ? feedback.feedback : feedback.feedback?.replace('Correct: ', '')}
                </p>
            </div>

            <button
                onClick={onNext}
                className="w-full py-6 mt-6 bg-slate-900 text-white rounded-[2rem] font-black text-2xl hover:bg-slate-800 transition-all flex items-center justify-center gap-4 group active:scale-[0.98] shadow-2xl"
            >
                NEXT TASK
                <Icon.ArrowRight size={32} className="group-hover:translate-x-2 transition-transform" />
            </button>
        </div>
    );
};